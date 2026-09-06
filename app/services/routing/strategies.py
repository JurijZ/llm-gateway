from abc import ABC, abstractmethod
from typing import List, Optional
from app.services.llm.base import LLMProvider
from app.core.models import get_model_cost

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
    def __init__(self):
        self.in_flight: dict = {}  # provider_name -> count

    def increment(self, provider_name: str):
        self.in_flight[provider_name] = self.in_flight.get(provider_name, 0) + 1

    def decrement(self, provider_name: str):
        if self.in_flight.get(provider_name, 0) > 0:
            self.in_flight[provider_name] -= 1

    def on_request_start(self, provider_name: str, model: Optional[str] = None) -> None:
        self.increment(provider_name)

    def on_request_end(self, provider_name: str, model: Optional[str] = None) -> None:
        self.decrement(provider_name)

    def select_provider(self, providers: List[LLMProvider], preference: Optional[str] = None) -> LLMProvider:
        # Returns the provider with the least in-flight requests
        return min(providers, key=lambda p: self.in_flight.get(p.get_provider_name(), 0))

class LatencyBasedStrategy(RoutingStrategy):
    def __init__(self):
        self.latencies: dict = {}  # provider_name -> rolling_avg
        self._rr_index: int = 0

    def update_latency(self, provider_name: str, latency: float):
        if provider_name not in self.latencies:
            self.latencies[provider_name] = latency
        else:
            # Exponential moving average
            self.latencies[provider_name] = self.latencies[provider_name] * 0.7 + latency * 0.3

    def on_first_chunk(self, provider_name: str, latency: float, model: Optional[str] = None) -> None:
        self.update_latency(provider_name, latency)

    def select_provider(self, providers: List[LLMProvider], preference: Optional[str] = None) -> LLMProvider:
        known = [p for p in providers if p.get_provider_name() in self.latencies]
        unknown = [p for p in providers if p.get_provider_name() not in self.latencies]

        # Round-robin unknown providers to gather latency data before committing
        if unknown:
            idx = self._rr_index % len(unknown)
            self._rr_index += 1
            return unknown[idx]

        return min(known, key=lambda p: self.latencies[p.get_provider_name()])

class CostLatencyTradeoffStrategy(RoutingStrategy):
    """
    Routes based on a composite score of latency, cost, and error rate.
    score = α × (1 / latency) + β × (1 / cost_per_token) + γ × (1 - error_rate)
    """
    def __init__(self, alpha: float = 0.4, beta: float = 0.4, gamma: float = 0.2):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.latencies: dict = {}     # provider_name -> rolling_avg
        self.costs: dict = {}         # provider_name -> cost_per_token
        self.error_rates: dict = {}   # provider_name -> rolling_avg
        self.epsilon = 1e-9

    def update_metrics(self, provider_name: str, latency: Optional[float] = None,
                       cost: Optional[float] = None, is_error: Optional[bool] = None):
        if latency is not None:
            if provider_name not in self.latencies:
                self.latencies[provider_name] = latency
            else:
                self.latencies[provider_name] = self.latencies[provider_name] * 0.7 + latency * 0.3

        if cost is not None:
            self.costs[provider_name] = cost

        if is_error is not None:
            error_val = 1.0 if is_error else 0.0
            if provider_name not in self.error_rates:
                self.error_rates[provider_name] = error_val
            else:
                self.error_rates[provider_name] = self.error_rates[provider_name] * 0.7 + error_val * 0.3

    def on_request_start(self, provider_name: str, model: Optional[str] = None) -> None:
        if provider_name not in self.costs:
            self.costs[provider_name] = get_model_cost(provider_name, model)

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

        # Exclude providers with 100% error rate when alternatives exist
        healthy = [p for p in providers if self.error_rates.get(p.get_provider_name(), 0.0) < 1.0]
        candidates = healthy if healthy else providers

        best_provider = candidates[0]
        max_score = -float('inf')

        for p in candidates:
            name = p.get_provider_name()

            latency = self.latencies.get(name, 1.0)
            cost = self.costs.get(name)
            if cost is None:
                cost = get_model_cost(name)
            error_rate = self.error_rates.get(name, 0.0)

            # score = α × (1 / latency) + β × (1 / cost_per_token) + γ × (1 - error_rate)
            score = (self.alpha * (1 / (latency + self.epsilon))) + \
                    (self.beta * (1 / (cost + self.epsilon))) + \
                    (self.gamma * (1 - error_rate))

            if score > max_score:
                max_score = score
                best_provider = p

        return best_provider
