from abc import ABC, abstractmethod
from typing import List, Optional
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

    @property
    def in_flight(self) -> dict:
        class InFlightProxy(dict):
            def __init__(self, store):
                self._store = store
            def get(self, key, default=0):
                val = self._store.get_in_flight(key)
                return val if val != 0 else default
            def __getitem__(self, key):
                return self._store.get_in_flight(key)
            def __setitem__(self, key, val):
                if hasattr(self._store, "_in_flight"):
                    self._store._in_flight[key] = val
        return InFlightProxy(self.store)

    def increment(self, provider_name: str):
        self.store.increment_in_flight(provider_name)

    def decrement(self, provider_name: str):
        self.store.decrement_in_flight(provider_name)

    def on_request_start(self, provider_name: str, model: Optional[str] = None) -> None:
        self.increment(provider_name)

    def on_request_end(self, provider_name: str, model: Optional[str] = None) -> None:
        self.decrement(provider_name)

    def select_provider(self, providers: List[LLMProvider], preference: Optional[str] = None) -> LLMProvider:
        return min(providers, key=lambda p: self.store.get_in_flight(p.get_provider_name()))

class LatencyBasedStrategy(RoutingStrategy):
    def __init__(self, store: Optional[MetricsStore] = None):
        self.store = store or InMemoryMetricsStore()
        self._rr_index: int = 0

    @property
    def latencies(self) -> dict:
        class LatenciesProxy(dict):
            def __init__(self, store):
                self._store = store
            def __contains__(self, key):
                return self._store.get_latency(key) is not None
            def __getitem__(self, key):
                val = self._store.get_latency(key)
                if val is None:
                    raise KeyError(key)
                return val
            def get(self, key, default=None):
                val = self._store.get_latency(key)
                return val if val is not None else default
            def __setitem__(self, key, val):
                if hasattr(self._store, "_latencies"):
                    self._store._latencies[key] = val
        return LatenciesProxy(self.store)

    def update_latency(self, provider_name: str, latency: float):
        self.store.update_latency(provider_name, latency)

    def on_first_chunk(self, provider_name: str, latency: float, model: Optional[str] = None) -> None:
        self.update_latency(provider_name, latency)

    def select_provider(self, providers: List[LLMProvider], preference: Optional[str] = None) -> LLMProvider:
        known = [p for p in providers if self.store.get_latency(p.get_provider_name()) is not None]
        unknown = [p for p in providers if self.store.get_latency(p.get_provider_name()) is None]

        if unknown:
            idx = self._rr_index % len(unknown)
            self._rr_index += 1
            return unknown[idx]

        return min(known, key=lambda p: self.store.get_latency(p.get_provider_name()))

class CostLatencyTradeoffStrategy(RoutingStrategy):
    """
    Routes based on a composite score of latency, cost, and error rate.
    score = α × (1 / latency) + β × (1 / cost_per_token) + γ × (1 - error_rate)
    """
    def __init__(self, alpha: float = 0.4, beta: float = 0.4, gamma: float = 0.2, store: Optional[MetricsStore] = None):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.store = store or InMemoryMetricsStore()
        self.epsilon = 1e-9


    @property
    def latencies(self) -> dict:
        class LatenciesProxy(dict):
            def __init__(self, store):
                self._store = store
            def __contains__(self, key):
                return self._store.get_latency(key) is not None
            def __getitem__(self, key):
                return self._store.get_latency(key)
            def get(self, key, default=None):
                v = self._store.get_latency(key)
                return v if v is not None else default
        return LatenciesProxy(self.store)

    @property
    def costs(self) -> dict:
        class CostsProxy(dict):
            def __init__(self, store):
                self._store = store
            def __contains__(self, key):
                return self._store.get_cost(key) is not None
            def __getitem__(self, key):
                return self._store.get_cost(key)
            def get(self, key, default=None):
                v = self._store.get_cost(key)
                return v if v is not None else default
            def __setitem__(self, key, val):
                self._store.set_cost(key, val)
        return CostsProxy(self.store)

    @property
    def error_rates(self) -> dict:
        class ErrorRatesProxy(dict):
            def __init__(self, store):
                self._store = store
            def __contains__(self, key):
                return True
            def __getitem__(self, key):
                return self._store.get_error_rate(key)
            def get(self, key, default=0.0):
                return self._store.get_error_rate(key)
        return ErrorRatesProxy(self.store)

    def update_metrics(self, provider_name: str, latency: Optional[float] = None,
                       cost: Optional[float] = None, is_error: Optional[bool] = None):
        if latency is not None:
            self.store.update_latency(provider_name, latency)
        if cost is not None:
            self.store.set_cost(provider_name, cost)
        if is_error is not None:
            self.store.update_error_rate(provider_name, is_error)

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
        if not providers:
            return None

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

