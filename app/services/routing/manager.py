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

from app.services.routing.store import get_metrics_store

@lru_cache(maxsize=None)
# ---------------------------------------------------------------------------
# §3.3 — Explicit strategy registry (replaces @lru_cache which cannot be
# invalidated in tests and is closed to extension).
# ---------------------------------------------------------------------------

_STRATEGY_REGISTRY: dict[str, RoutingStrategy] = {}


def _init_registry() -> None:
    store = get_metrics_store()
    _STRATEGY_REGISTRY.update({
        "hardcoded":    HardcodedStrategy(),
        "load_balance": LeastInFlightStrategy(store),
        "latency":      LatencyBasedStrategy(store),
        "cost_latency": CostLatencyTradeoffStrategy(store=store),
    })


def get_strategy(strategy_name: str) -> RoutingStrategy:
    if strategy_name == "load_balance":
        return LeastInFlightStrategy(get_metrics_store())
    elif strategy_name == "latency":
        return LatencyBasedStrategy(get_metrics_store())
    elif strategy_name == "cost_latency":
        return CostLatencyTradeoffStrategy(store=get_metrics_store())
    else:
        return HardcodedStrategy()
    if not _STRATEGY_REGISTRY:
        _init_registry()
    return _STRATEGY_REGISTRY.get(strategy_name, _STRATEGY_REGISTRY["hardcoded"])



from app.services.routing.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
    CircuitBreakerOpenError,
    get_circuit_breaker,
)
from app.services.telemetry import get_telemetry

class RouterManager:
    def __init__(self, providers: List[LLMProvider], circuit_breaker: Optional[CircuitBreaker] = None):
        self.providers = providers
        self.circuit_breaker = circuit_breaker or get_circuit_breaker()
        # §3.4: last_selected_provider / last_selected_model are set as a side-effect
        # when the first chunk is committed. Each RouterManager instance is created
        # per-request (see chat.py get_router_manager), so there is no cross-request
        # race. Fields exist solely so chat.py can read provider/model for HTTP headers
        # after stream completion.
        self.last_selected_provider: Optional[str] = None
        self.last_selected_model: Optional[str] = None

    # ------------------------------------------------------------------
    # Provider / candidate selection
    # ------------------------------------------------------------------

    def select_provider(self, active_strategy: RoutingStrategy, preference: Optional[str] = None) -> LLMProvider:
        """
        Public helper: returns the single provider the active strategy selects.
        For HardcodedStrategy the model preference is used to resolve a provider;
        for all other strategies the preference is ignored (strategy overrides).
        """
        resolved_provider_name, _ = (
            get_model_info(preference) if preference else (None, None)
        )
        return active_strategy.select_provider(
            self.providers, preference=resolved_provider_name or preference
        )

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

        Duplicate (provider_name, model) pairs are removed after the chain is
        assembled (§3.5) to avoid redundant retry attempts.
        """
        resolved_provider_name, resolved_model_id = (
            get_model_info(preference) if preference else (None, None)
        )

        # 1. Primary: strategy picks the provider.
        primary = active_strategy.select_provider(
            self.providers, preference=resolved_provider_name or preference
        )

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

        # §3.5: Deduplicate — remove duplicate (provider_name, model) pairs while
        # preserving order. Duplicates arise when fallback_models repeats a model.
        seen: set[tuple[str, Optional[str]]] = set()
        deduped: List[Tuple[LLMProvider, Optional[str]]] = []
        for p, m in candidates:
            key = (p.get_provider_name(), m)
            if key not in seen:
                seen.add(key)
                deduped.append((p, m))
        candidates = deduped

        # Reorder candidates so providers with OPEN circuits are deprioritized
        # when healthy alternatives exist in the candidate chain.
        candidates.sort(
            key=lambda c: 0 if self.circuit_breaker.get_state(c[0].get_provider_name()) != CircuitState.OPEN else 1
        )

        return candidates

    # ------------------------------------------------------------------
    # Streaming with two-phase timeouts
    # ------------------------------------------------------------------

    async def _stream_with_timeouts(
        self,
        provider: LLMProvider,
        messages: List[dict],
        model: Optional[str],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
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
        kwargs = {}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if top_p is not None:
            kwargs["top_p"] = top_p

        try:
            aiter = provider.stream_chat(
                messages,
                model=model,
                **kwargs,
            ).__aiter__()
        except TypeError:
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
            # §2.3: Always close the underlying stream. Re-raise CancelledError and
            # GeneratorExit so that cancellation propagates correctly — only swallow
            # mundane exceptions from close().
            try:
                await aiter.aclose()
            except (asyncio.CancelledError, GeneratorExit):
                raise  # let cancellation / generator teardown propagate
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Metric helpers (delegates to active_strategy hooks for backward compatibility)
    # ------------------------------------------------------------------

    def _record_latency(self, active_strategy: RoutingStrategy, provider_name: str, latency: float):
        active_strategy.on_first_chunk(provider_name, latency)

    def _record_success(self, active_strategy: RoutingStrategy, provider_name: str):
        active_strategy.on_request_success(provider_name)

    def _record_error(self, active_strategy: RoutingStrategy, provider_name: str):
        active_strategy.on_request_error(provider_name, Exception("provider failed"))

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def stream_with_fallback(
        self,
        messages: List[dict],
        preference: Optional[str] = None,
        fallback_models: Optional[List[str]] = None,
        strategy_type: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
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

        # §4.4: Structured routing decision log — queryable as individual fields
        # in JSON log aggregators rather than an opaque f-string.
        logger.info(
            f"Strategy: {type(active_strategy).__name__} | "
            f"Candidates: {[(p.get_provider_name(), m) for p, m in candidates]}"
            "Routing decision",
            extra={
                "strategy": type(active_strategy).__name__,
                "candidates": [
                    {"provider": p.get_provider_name(), "model": m}
                    for p, m in candidates
                ],
            },
        )

        last_error = None
        prev_provider_name: Optional[str] = None
        for idx, (provider, target_model) in enumerate(candidates):
            provider_name = provider.get_provider_name()

            # Fail-fast circuit breaker check:
            # If circuit is OPEN, bypass candidate immediately without waiting for TTFC_TIMEOUT.
            if not self.circuit_breaker.can_execute(provider_name):
                logger.warning(
                    f"Circuit breaker for provider '{provider_name}' is OPEN. "
                    f"Bypassing immediately without timeout."
                )
                continue

            if idx > 0:
                get_telemetry().record_fallback()
                # §4.3: Record structured fallback event with context for debugging.
                get_telemetry().record_fallback(
                    failed_provider=prev_provider_name or "unknown",
                    reason=str(last_error) if last_error else "circuit_open",
                    next_provider=provider_name,
                )

            logger.info(f"Trying {provider_name} model={target_model or 'default'}")

            # §2.2: on_request_start MUST always be paired with on_request_end in
            # the finally block below, regardless of any exception type (timeout,
            # RuntimeError, StopAsyncIteration, CancelledError).
            active_strategy.on_request_start(provider_name, target_model)

            start_time = asyncio.get_running_loop().time()
            committed = False  # True once the first chunk is yielded upstream

            try:
                async for chunk in self._stream_with_timeouts(
                    provider,
                    messages,
                    target_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                ):
                    if not committed:
                        latency = asyncio.get_running_loop().time() - start_time
                        active_strategy.on_first_chunk(provider_name, latency, target_model)
                        get_telemetry().record_ttfc(latency * 1000, provider=provider_name, model=target_model)
                        self.last_selected_provider = provider_name
                        self.last_selected_model = target_model
                        committed = True
                    yield chunk

                active_strategy.on_request_success(provider_name, target_model)
                self.circuit_breaker.record_success(provider_name)
                duration_ms = (asyncio.get_running_loop().time() - start_time) * 1000
                get_telemetry().record_request_complete(
                    duration_ms, success=True, provider=provider_name, model=target_model
                )
                return  # clean exit

            except Exception as e:
                logger.warning(
                    f"Provider {provider_name} failed "
                    f"({'committed' if committed else 'before first chunk'}): {e}"
                )
                active_strategy.on_request_error(provider_name, e, target_model)
                self.circuit_breaker.record_failure(provider_name)
                last_error = e
                prev_provider_name = provider_name

                if committed:
                    # Bytes already sent — surface the error immediately.
                    raise

                # Not yet committed — try next candidate.
                logger.info(f"Trying next candidate for {provider_name}")
                continue

            finally:
                # §2.2: Invariant — on_request_end MUST be called here regardless
                # of how the try block exits (success, exception, or return).
                active_strategy.on_request_end(provider_name, target_model)

        if last_error:
            raise last_error
        raise CircuitBreakerOpenError(
            "All candidate upstream providers are currently unavailable due to open circuit breakers"
        )

