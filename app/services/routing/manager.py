from typing import List, Optional, AsyncGenerator
import asyncio
from app.services.llm.base import LLMProvider
from app.services.routing.strategies import RoutingStrategy, HardcodedStrategy, LeastInFlightStrategy, LatencyBasedStrategy, CostLatencyTradeoffStrategy
from app.core.config import settings
from app.core.models import get_model_info
import logging

logger = logging.getLogger(__name__)

class RouterManager:
    def __init__(self, providers: List[LLMProvider], strategy_type: str = None):
        self.providers = providers
        self.strategy: RoutingStrategy = self._get_strategy(strategy_type or settings.DEFAULT_STRATEGY)

    def _get_strategy(self, strategy_name: str) -> RoutingStrategy:
        if strategy_name == "load_balance":
            return LeastInFlightStrategy()
        elif strategy_name == "latency":
            return LatencyBasedStrategy()
        elif strategy_name == "cost_latency":
            return CostLatencyTradeoffStrategy()
        else:
            return HardcodedStrategy()

    def select_provider(self, preference: Optional[str] = None) -> LLMProvider:
        # For HardcodedStrategy, resolve the model preference to a provider name so the
        # caller's model hint is honoured. For all other strategies the strategy itself
        # decides which provider to use (strategy overrides model selection).
        if isinstance(self.strategy, HardcodedStrategy) and preference:
            resolved_provider_name, _ = get_model_info(preference)
            return self.strategy.select_provider(self.providers, resolved_provider_name or preference)
        return self.strategy.select_provider(self.providers)

    async def stream_with_fallback(self, messages: List[dict], preference: Optional[str] = None) -> AsyncGenerator[str, None]:
        # Resolve the model preference to (provider, model_id) for use in target_model below.
        # This resolution is independent of which provider the strategy selects.
        resolved_provider_name, resolved_model_id = get_model_info(preference) if preference else (None, None)

        # Strategy selects the primary provider.
        # Non-hardcoded strategies ignore preference entirely (strategy overrides model selection).
        if isinstance(self.strategy, HardcodedStrategy):
            selected = self.strategy.select_provider(self.providers, resolved_provider_name or preference)
        else:
            selected = self.strategy.select_provider(self.providers)

        others = [p for p in self.providers if p != selected]
        candidates = [selected] + others

        logger.info(f"Selected routing strategy: {type(self.strategy).__name__}")

        last_error = None
        for provider in candidates:
            provider_name = provider.get_provider_name()

            # Use the resolved model only when this provider is the one the model belongs to.
            # When the strategy picks a different provider, that provider uses its own default.
            target_model = resolved_model_id if provider_name == resolved_provider_name else None

            logger.info(f"Trying provider {provider_name} with model {target_model or 'default'}")

            if isinstance(self.strategy, LeastInFlightStrategy):
                self.strategy.increment(provider_name)

            start_time = asyncio.get_event_loop().time()
            try:
                async with asyncio.timeout(settings.REQUEST_TIMEOUT):
                    first_chunk = True
                    async for chunk in provider.stream_chat(messages, model=target_model):
                        if first_chunk:
                            latency = asyncio.get_event_loop().time() - start_time
                            if isinstance(self.strategy, LatencyBasedStrategy):
                                self.strategy.update_latency(provider_name, latency)
                            elif isinstance(self.strategy, CostLatencyTradeoffStrategy):
                                self.strategy.update_metrics(provider_name, latency=latency)
                            first_chunk = False
                        yield chunk

                if isinstance(self.strategy, CostLatencyTradeoffStrategy):
                    self.strategy.update_metrics(provider_name, is_error=False)
                return  # Success

            except (asyncio.TimeoutError, Exception) as e:
                logger.warning(f"Provider {provider_name} failed: {str(e)}")
                if isinstance(self.strategy, CostLatencyTradeoffStrategy):
                    self.strategy.update_metrics(provider_name, is_error=True)
                last_error = e
                continue

            finally:
                if isinstance(self.strategy, LeastInFlightStrategy):
                    self.strategy.decrement(provider_name)

        if last_error:
            raise last_error
