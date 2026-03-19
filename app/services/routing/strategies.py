from abc import ABC, abstractmethod
from typing import List, Optional
from app.services.llm.base import LLMProvider

class RoutingStrategy(ABC):
    @abstractmethod
    def select_provider(self, providers: List[LLMProvider], preference: Optional[str] = None) -> LLMProvider:
        """Select a provider from the list based on the strategy logic."""
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
            cost = self.costs.get(name, 0.001)
            error_rate = self.error_rates.get(name, 0.0)

            # score = α × (1 / latency) + β × (1 / cost_per_token) + γ × (1 - error_rate)
            score = (self.alpha * (1 / (latency + self.epsilon))) + \
                    (self.beta * (1 / (cost + self.epsilon))) + \
                    (self.gamma * (1 - error_rate))

            if score > max_score:
                max_score = score
                best_provider = p

        return best_provider
