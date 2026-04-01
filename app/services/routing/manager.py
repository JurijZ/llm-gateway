from typing import List, Optional, AsyncGenerator, Tuple
import asyncio
from app.services.llm.base import LLMProvider
from app.services.routing.strategies import (
    RoutingStrategy, HardcodedStrategy, LeastInFlightStrategy,
    LatencyBasedStrategy, CostLatencyTradeoffStrategy,
)
from app.core.config import settings
from app.core.models import get_model_info
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

@lru_cache(maxsize=None)
def get_strategy(strategy_name: str) -> RoutingStrategy:
    if strategy_name == "load_balance":
        return LeastInFlightStrategy()
    elif strategy_name == "latency":
        return LatencyBasedStrategy()
    elif strategy_name == "cost_latency":
        return CostLatencyTradeoffStrategy()
    else:
        return HardcodedStrategy()


class RouterManager:
    def __init__(self, providers: List[LLMProvider]):
        self.providers = providers

    # ------------------------------------------------------------------
    # Provider / candidate selection
    # ------------------------------------------------------------------

    def select_provider(self, active_strategy: RoutingStrategy, preference: Optional[str] = None) -> LLMProvider:
        """
        Public helper: returns the single provider the active strategy selects.
        For HardcodedStrategy the model preference is used to resolve a provider;
        for all other strategies the preference is ignored (strategy overrides).
        """
        if isinstance(active_strategy, HardcodedStrategy) and preference:
            resolved_provider_name, _ = get_model_info(preference)
            return active_strategy.select_provider(
                self.providers, resolved_provider_name or preference
            )
        return active_strategy.select_provider(self.providers)

    def _build_candidates(
        self,
        active_strategy: RoutingStrategy,
        preference: Optional[str],
        fallback_models: Optional[List[str]],
    ) -> List[Tuple[LLMProvider, Optional[str]]]:
        """
        Returns an ordered list of (provider, model_id) pairs to attempt,
        falling through on each failure.

        Order:
          1. Strategy-selected provider with the resolved model for that provider.
          2. Explicit fallback_models list (resolved in declared order).
             The same provider may appear more than once with different models.
          3. Any remaining configured providers not yet in the chain, at their
             default model, as a last-resort safety net.
        """
        resolved_provider_name, resolved_model_id = (
            get_model_info(preference) if preference else (None, None)
        )

        # 1. Primary: strategy picks the provider.
        if isinstance(active_strategy, HardcodedStrategy):
            primary = active_strategy.select_provider(
                self.providers, resolved_provider_name or preference
            )
        else:
            primary = active_strategy.select_provider(self.providers)

        primary_model = (
            resolved_model_id
            if primary.get_provider_name() == resolved_provider_name
            else None
        )
        candidates: List[Tuple[LLMProvider, Optional[str]]] = [(primary, primary_model)]

        # 2. Explicit fallback chain.
        if fallback_models:
            for fb_name in fallback_models:
                fb_provider_name, fb_model_id = get_model_info(fb_name)
                if not fb_provider_name:
                    logger.warning(f"Unknown fallback model '{fb_name}', skipping.")
                    continue
                fb_provider = next(
                    (p for p in self.providers if p.get_provider_name() == fb_provider_name),
                    None,
                )
                if not fb_provider:
                    logger.warning(
                        f"Provider '{fb_provider_name}' for fallback '{fb_name}' "
                        f"is not configured, skipping."
                    )
                    continue
                candidates.append((fb_provider, fb_model_id))

        # 3. Last-resort: any provider not yet covered by any candidate entry.
        providers_in_chain = {p.get_provider_name() for p, _ in candidates}
        for p in self.providers:
            if p.get_provider_name() not in providers_in_chain:
                candidates.append((p, None))

        return candidates

    # ------------------------------------------------------------------
    # Streaming with two-phase timeouts
    # ------------------------------------------------------------------

    async def _stream_with_timeouts(
        self,
        provider: LLMProvider,
        messages: List[dict],
        model: Optional[str],
    ) -> AsyncGenerator[str, None]:
        """
        Wraps provider.stream_chat with two-phase streaming timeouts:

        Phase 1 — TTFC (time-to-first-chunk):
            asyncio.TimeoutError is raised before any chunk is yielded.
            The caller has not yet committed any bytes, so fallback is safe.

        Phase 2 — idle (between-chunk) timeout:
            The deadline is reset after every chunk.  asyncio.TimeoutError
            raised here means the stream stalled mid-response; the caller
            has already committed bytes to the HTTP client, so no fallback
            is possible and the error propagates to the client.
        """
        aiter = provider.stream_chat(messages, model=model).__aiter__()
        try:
            # --- Phase 1: wait for first chunk ---
            async with asyncio.timeout(settings.TTFC_TIMEOUT):
                try:
                    first = await aiter.__anext__()
                except StopAsyncIteration:
                    return
            yield first

            # --- Phase 2: subsequent chunks with idle timeout ---
            loop = asyncio.get_running_loop()
            async with asyncio.timeout(settings.CHUNK_TIMEOUT) as deadline:
                while True:
                    try:
                        chunk = await aiter.__anext__()
                        # Reset the deadline after each successful chunk.
                        deadline.reschedule(loop.time() + settings.CHUNK_TIMEOUT)
                        yield chunk
                    except StopAsyncIteration:
                        break
        finally:
            # Always close the underlying stream (handles cancellation / timeout).
            try:
                await aiter.aclose()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Metric helpers
    # ------------------------------------------------------------------

    def _record_latency(self, active_strategy: RoutingStrategy, provider_name: str, latency: float):
        if isinstance(active_strategy, LatencyBasedStrategy):
            active_strategy.update_latency(provider_name, latency)
        elif isinstance(active_strategy, CostLatencyTradeoffStrategy):
            active_strategy.update_metrics(provider_name, latency=latency)

    def _record_success(self, active_strategy: RoutingStrategy, provider_name: str):
        if isinstance(active_strategy, CostLatencyTradeoffStrategy):
            active_strategy.update_metrics(provider_name, is_error=False)

    def _record_error(self, active_strategy: RoutingStrategy, provider_name: str):
        if isinstance(active_strategy, CostLatencyTradeoffStrategy):
            active_strategy.update_metrics(provider_name, is_error=True)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def stream_with_fallback(
        self,
        messages: List[dict],
        preference: Optional[str] = None,
        fallback_models: Optional[List[str]] = None,
        strategy_type: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Streams a response, falling back through the candidate list on failure.

        Fallback is only attempted when no bytes have been committed to the
        caller yet (i.e. the failure occurred before the first chunk was
        yielded).  Once committed, errors propagate immediately.

        Candidate order is determined by _build_candidates():
          routing-strategy primary → explicit fallback_models → remaining providers.
        """
        active_strategy = get_strategy(strategy_type or settings.DEFAULT_STRATEGY)
        candidates = self._build_candidates(active_strategy, preference, fallback_models)
        logger.info(
            f"Strategy: {type(active_strategy).__name__} | "
            f"Candidates: {[(p.get_provider_name(), m) for p, m in candidates]}"
        )

        last_error = None
        for provider, target_model in candidates:
            provider_name = provider.get_provider_name()
            logger.info(f"Trying {provider_name} model={target_model or 'default'}")

            if isinstance(active_strategy, LeastInFlightStrategy):
                active_strategy.increment(provider_name)

            start_time = asyncio.get_running_loop().time()
            committed = False  # True once the first chunk is yielded upstream

            try:
                async for chunk in self._stream_with_timeouts(provider, messages, target_model):
                    if not committed:
                        latency = asyncio.get_running_loop().time() - start_time
                        self._record_latency(active_strategy, provider_name, latency)
                        committed = True
                    yield chunk

                self._record_success(active_strategy, provider_name)
                return  # clean exit

            except Exception as e:
                logger.warning(
                    f"Provider {provider_name} failed "
                    f"({'committed' if committed else 'before first chunk'}): {e}"
                )
                self._record_error(active_strategy, provider_name)
                last_error = e

                if committed:
                    # Bytes already sent — surface the error immediately.
                    raise

                # Not yet committed — try next candidate.
                logger.info(f"Trying next candidate for {provider_name}")
                continue

            finally:
                if isinstance(active_strategy, LeastInFlightStrategy):
                    active_strategy.decrement(provider_name)

        if last_error:
            raise last_error
