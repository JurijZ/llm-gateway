from abc import ABC, abstractmethod
from typing import List, Optional
import threading
from app.services.llm.base import LLMProvider
from app.core.models import get_model_cost
from app.services.routing.store import MetricsStore, InMemoryMetricsStore, get_metrics_store


class RoutingStrategy(ABC):
    @abstractmethod
    def select_provider(self, providers: List[LLMProvider], preference: Optional[str] = None) -> LLMProvider:
        """Select a provider from the list based on the strategy logic."""
        pass

    def on_request_start(self, provider_name: str, model: Optional[str] = None) -> None:
        """Hook called when a request begins attempting with a provider."""
        pass

    def on_first_chunk(self, provider_name: str, latency: float, model: Optional[str] = None) -> None:
        """Hook called when the first chunk is received (measuring TTFC)."""
        pass

    def on_request_success(self, provider_name: str, model: Optional[str] = None, **kwargs) -> None:
        """Hook called when a request completes successfully."""
        pass

    def on_request_error(self, provider_name: str, error: Exception, model: Optional[str] = None, **kwargs) -> None:
        """Hook called when a provider fails."""
        pass

    def on_request_end(self, provider_name: str, model: Optional[str] = None) -> None:
        """Hook called in finally block after provider attempt finishes."""
        pass


class HardcodedStrategy(RoutingStrategy):
    def select_provider(self, providers: List[LLMProvider], preference: Optional[str] = None) -> LLMProvider:
        # Returns the first provider in the list, or one matching the preference
        if preference:
            for p in providers:
                if p.get_provider_name() == preference:
                    return p
        return providers[0]


class LeastInFlightStrategy(RoutingStrategy):
    def __init__(self, store: Optional[MetricsStore] = None):
        self.store = store or InMemoryMetricsStore()

    def get_in_flight(self, provider_name: str) -> int:
        """Return the current in-flight request count for a provider."""
        return self.store.get_in_flight(provider_name)

    def on_request_start(self, provider_name: str, model: Optional[str] = None) -> None:
        self.store.increment_in_flight(provider_name)

    def on_request_end(self, provider_name: str, model: Optional[str] = None) -> None:
        self.store.decrement_in_flight(provider_name)

    def select_provider(self, providers: List[LLMProvider], preference: Optional[str] = None) -> LLMProvider:
        return min(providers, key=lambda p: self.store.get_in_flight(p.get_provider_name()))


class _LatencyAwareStrategy(RoutingStrategy):
    """
    Base class for strategies that track EMA latency per provider.
    Provides shared update_latency() and on_first_chunk() so that
    LatencyBasedStrategy and CostLatencyTradeoffStrategy don't duplicate logic.
    """
    def __init__(self, store: Optional[MetricsStore] = None):
        self.store = store or InMemoryMetricsStore()

    def update_latency(self, provider_name: str, latency: float) -> None:
        """Update the EMA latency for a provider."""
        self.store.update_latency(provider_name, latency)

    def get_latency(self, provider_name: str) -> Optional[float]:
        """Return the current EMA latency for a provider, or None if unknown."""
        return self.store.get_latency(provider_name)

    def on_first_chunk(self, provider_name: str, latency: float, model: Optional[str] = None) -> None:
        self.update_latency(provider_name, latency)


class LatencyBasedStrategy(_LatencyAwareStrategy):
    def __init__(self, store: Optional[MetricsStore] = None):
        super().__init__(store)
        # §5.3: protect _rr_index with a lock so concurrent requests both
        # explore different unknown providers rather than both picking the same one.
        self._rr_lock = threading.Lock()
        self._rr_index: int = 0

    def select_provider(self, providers: List[LLMProvider], preference: Optional[str] = None) -> LLMProvider:
        known = [p for p in providers if self.store.get_latency(p.get_provider_name()) is not None]
        unknown = [p for p in providers if self.store.get_latency(p.get_provider_name()) is None]

        if unknown:
            with self._rr_lock:
                idx = self._rr_index % len(unknown)
                self._rr_index += 1
            return unknown[idx]

        return min(known, key=lambda p: self.store.get_latency(p.get_provider_name()))


class CostLatencyTradeoffStrategy(_LatencyAwareStrategy):
    """
    Routes based on a composite score of latency, cost, and error rate.
    score = α × (1 / latency) + β × (1 / cost_per_token) + γ × (1 - error_rate)
    """
    def __init__(self, alpha: float = 0.4, beta: float = 0.4, gamma: float = 0.2, store: Optional[MetricsStore] = None):
        super().__init__(store)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.epsilon = 1e-9

    def update_metrics(self, provider_name: str, latency: Optional[float] = None,
                       cost: Optional[float] = None, is_error: Optional[bool] = None):
        if latency is not None:
            self.store.update_latency(provider_name, latency)
        if cost is not None:
            self.store.set_cost(provider_name, cost)
        if is_error is not None:
            self.store.update_error_rate(provider_name, is_error)

    def get_cost(self, provider_name: str) -> Optional[float]:
        """Return the current cost for a provider, or None if unknown."""
        return self.store.get_cost(provider_name)

    def get_error_rate(self, provider_name: str) -> float:
        """Return the current EMA error rate for a provider."""
        return self.store.get_error_rate(provider_name)

    def on_request_start(self, provider_name: str, model: Optional[str] = None) -> None:
        if self.store.get_cost(provider_name) is None:
            self.store.set_cost(provider_name, get_model_cost(provider_name, model))

    def on_first_chunk(self, provider_name: str, latency: float, model: Optional[str] = None) -> None:
        cost = get_model_cost(provider_name, model) if model else None
        self.update_metrics(provider_name, latency=latency, cost=cost)

    def on_request_success(self, provider_name: str, model: Optional[str] = None, **kwargs) -> None:
        self.update_metrics(provider_name, is_error=False)

    def on_request_error(self, provider_name: str, error: Exception, model: Optional[str] = None, **kwargs) -> None:
        self.update_metrics(provider_name, is_error=True)

    def select_provider(self, providers: List[LLMProvider], preference: Optional[str] = None) -> LLMProvider:
        # §2.4: fail fast instead of silently returning None (type violation)
        if not providers:
            raise ValueError("No providers available for CostLatencyTradeoffStrategy.select_provider")

        healthy = [p for p in providers if self.store.get_error_rate(p.get_provider_name()) < 1.0]
        candidates = healthy if healthy else providers

        best_provider = candidates[0]
        max_score = -float('inf')

        for p in candidates:
            name = p.get_provider_name()

            latency = self.store.get_latency(name)
            if latency is None:
                latency = 1.0
            cost = self.store.get_cost(name)
            if cost is None:
                cost = get_model_cost(name)
            error_rate = self.store.get_error_rate(name)

            # score = α × (1 / latency) + β × (1 / cost_per_token) + γ × (1 - error_rate)
            score = (self.alpha * (1 / (latency + self.epsilon))) + \
                    (self.beta * (1 / (cost + self.epsilon))) + \
                    (self.gamma * (1 - error_rate))

            if score > max_score:
                max_score = score
                best_provider = p

        return best_provider
