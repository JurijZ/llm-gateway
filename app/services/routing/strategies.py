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

# Placeholder for future strategies
class LeastInFlightStrategy(RoutingStrategy):
    def __init__(self):
        self.in_flight = {} # provider_name -> count

    def select_provider(self, providers: List[LLMProvider], preference: Optional[str] = None) -> LLMProvider:
        # Returns the provider with the least in-flight requests
        # Simple implementation for now
        return providers[0]

class LatencyBasedStrategy(RoutingStrategy):
    def __init__(self):
        self.latencies = {} # provider_name -> rolling_avg

    def update_latency(self, provider_name: str, latency: float):
        if provider_name not in self.latencies:
            self.latencies[provider_name] = latency
        else:
            # Simple rolling average
            self.latencies[provider_name] = self.latencies[provider_name] * 0.7 + latency * 0.3

    def select_provider(self, providers: List[LLMProvider], preference: Optional[str] = None) -> LLMProvider:
        if not self.latencies:
            return providers[0]
            
        # Select provider with lowest latency
        best_provider = providers[0]
        min_latency = self.latencies.get(best_provider.get_provider_name(), float('inf'))
        
        for p in providers[1:]:
            lat = self.latencies.get(p.get_provider_name(), float('inf'))
            if lat < min_latency:
                min_latency = lat
                best_provider = p
                
        return best_provider
