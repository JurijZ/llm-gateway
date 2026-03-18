from typing import List, Optional, AsyncGenerator
import asyncio
from app.services.llm.base import LLMProvider
from app.services.routing.strategies import RoutingStrategy, HardcodedStrategy, LeastInFlightStrategy, LatencyBasedStrategy
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
        else:
            return HardcodedStrategy()

    def select_provider(self, preference: Optional[str] = None) -> LLMProvider:
        # If preference is provided, check if it maps to a specific provider
        resolved_provider_name = None
        if preference:
            resolved_provider_name, _ = get_model_info(preference)
            
        return self.strategy.select_provider(self.providers, resolved_provider_name or preference)

    async def stream_with_fallback(self, messages: List[dict], preference: Optional[str] = None) -> AsyncGenerator[str, None]:
        # Resolve preference to actual provider and model
        resolved_provider_name = None
        resolved_model_id = None
        if preference:
            resolved_provider_name, resolved_model_id = get_model_info(preference)
        
        # Sort providers: selected one first, then others
        # Pass resolved_provider_name if we have it, else the raw preference
        selected = self.select_provider(resolved_provider_name or preference)
        others = [p for p in self.providers if p != selected]
        candidates = [selected] + others
        
        last_error = None
        for provider in candidates:
            # If this candidate IS the resolved provider, use the resolved model ID
            # Otherwise, use None (which uses provider's default)
            target_model = resolved_model_id if provider.get_provider_name() == resolved_provider_name else None

            print(f"Trying provider {provider.get_provider_name()} with model {target_model or 'default'}")
            
            start_time = asyncio.get_event_loop().time()
            try:
                logger.info(f"Attempting stream with provider: {provider.get_provider_name()} (model: {target_model or 'default'})")
                # Apply timeout
                async with asyncio.timeout(settings.REQUEST_TIMEOUT):
                    first_chunk = True
                    async for chunk in provider.stream_chat(messages, model=target_model):
                        if first_chunk:
                            # Record latency to first chunk
                            latency = asyncio.get_event_loop().time() - start_time
                            if isinstance(self.strategy, LatencyBasedStrategy):
                                self.strategy.update_latency(provider.get_provider_name(), latency)
                            first_chunk = False
                        yield chunk
                return # Success
            except (asyncio.TimeoutError, Exception) as e:
                logger.warning(f"Provider {provider.get_provider_name()} failed: {str(e)}")
                last_error = e
                continue
        
        if last_error:
            raise last_error
