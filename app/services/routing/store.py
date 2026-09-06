from abc import ABC, abstractmethod
from typing import Optional, Dict
import threading
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)

class MetricsStore(ABC):
    """Abstract interface for storing routing metrics across requests and workers."""

    @abstractmethod
    def get_in_flight(self, provider_name: str) -> int:
        pass

    @abstractmethod
    def increment_in_flight(self, provider_name: str) -> int:
        pass

    @abstractmethod
    def decrement_in_flight(self, provider_name: str) -> int:
        pass

    @abstractmethod
    def get_latency(self, provider_name: str) -> Optional[float]:
        pass

    @abstractmethod
    def update_latency(self, provider_name: str, latency: float, weight_old: float = 0.7) -> float:
        pass

    @abstractmethod
    def get_error_rate(self, provider_name: str) -> float:
        pass

    @abstractmethod
    def update_error_rate(self, provider_name: str, is_error: bool, weight_old: float = 0.7) -> float:
        pass

    @abstractmethod
    def get_cost(self, provider_name: str) -> Optional[float]:
        pass

    @abstractmethod
    def set_cost(self, provider_name: str, cost: float) -> None:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass


class InMemoryMetricsStore(MetricsStore):
    """
    Thread-safe in-memory store for single-process and test environments.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._in_flight: Dict[str, int] = {}
        self._latencies: Dict[str, float] = {}
        self._error_rates: Dict[str, float] = {}
        self._costs: Dict[str, float] = {}

    def get_in_flight(self, provider_name: str) -> int:
        with self._lock:
            return self._in_flight.get(provider_name, 0)

    def increment_in_flight(self, provider_name: str) -> int:
        with self._lock:
            val = self._in_flight.get(provider_name, 0) + 1
            self._in_flight[provider_name] = val
            return val

    def decrement_in_flight(self, provider_name: str) -> int:
        with self._lock:
            val = max(0, self._in_flight.get(provider_name, 0) - 1)
            self._in_flight[provider_name] = val
            return val

    def get_latency(self, provider_name: str) -> Optional[float]:
        with self._lock:
            return self._latencies.get(provider_name)

    def update_latency(self, provider_name: str, latency: float, weight_old: float = 0.7) -> float:
        with self._lock:
            if provider_name not in self._latencies:
                val = latency
            else:
                val = self._latencies[provider_name] * weight_old + latency * (1.0 - weight_old)
            self._latencies[provider_name] = val
            return val

    def get_error_rate(self, provider_name: str) -> float:
        with self._lock:
            return self._error_rates.get(provider_name, 0.0)

    def update_error_rate(self, provider_name: str, is_error: bool, weight_old: float = 0.7) -> float:
        with self._lock:
            err_val = 1.0 if is_error else 0.0
            if provider_name not in self._error_rates:
                val = err_val
            else:
                val = self._error_rates[provider_name] * weight_old + err_val * (1.0 - weight_old)
            self._error_rates[provider_name] = val
            return val

    def get_cost(self, provider_name: str) -> Optional[float]:
        with self._lock:
            return self._costs.get(provider_name)

    def set_cost(self, provider_name: str, cost: float) -> None:
        with self._lock:
            self._costs[provider_name] = cost

    def reset(self) -> None:
        with self._lock:
            self._in_flight.clear()
            self._latencies.clear()
            self._error_rates.clear()
            self._costs.clear()


class RedisMetricsStore(MetricsStore):
    """
    Distributed metrics store utilizing Redis for multi-worker and multi-container setups.
    Falls back gracefully to in-memory store if Redis is unavailable.
    """
    def __init__(self, redis_url: str):
        self._fallback = InMemoryMetricsStore()
        self.redis_url = redis_url
        self._client = None
        try:
            import redis
            self._client = redis.Redis.from_url(redis_url, decode_responses=True)
            self._client.ping()
            logger.info(f"Connected to Redis at {redis_url}")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis ({e}). Falling back to InMemoryMetricsStore.")
            self._client = None

    def get_in_flight(self, provider_name: str) -> int:
        if not self._client:
            return self._fallback.get_in_flight(provider_name)
        try:
            val = self._client.get(f"llm:in_flight:{provider_name}")
            return int(val) if val else 0
        except Exception:
            return self._fallback.get_in_flight(provider_name)

    def increment_in_flight(self, provider_name: str) -> int:
        if not self._client:
            return self._fallback.increment_in_flight(provider_name)
        try:
            return self._client.incr(f"llm:in_flight:{provider_name}")
        except Exception:
            return self._fallback.increment_in_flight(provider_name)

    def decrement_in_flight(self, provider_name: str) -> int:
        if not self._client:
            return self._fallback.decrement_in_flight(provider_name)
        try:
            val = self._client.decr(f"llm:in_flight:{provider_name}")
            if val < 0:
                self._client.set(f"llm:in_flight:{provider_name}", 0)
                return 0
            return val
        except Exception:
            return self._fallback.decrement_in_flight(provider_name)

    def get_latency(self, provider_name: str) -> Optional[float]:
        if not self._client:
            return self._fallback.get_latency(provider_name)
        try:
            val = self._client.get(f"llm:latency:{provider_name}")
            return float(val) if val is not None else None
        except Exception:
            return self._fallback.get_latency(provider_name)

    def update_latency(self, provider_name: str, latency: float, weight_old: float = 0.7) -> float:
        if not self._client:
            return self._fallback.update_latency(provider_name, latency, weight_old)
        try:
            current = self.get_latency(provider_name)
            new_val = latency if current is None else current * weight_old + latency * (1.0 - weight_old)
            self._client.set(f"llm:latency:{provider_name}", new_val)
            return new_val
        except Exception:
            return self._fallback.update_latency(provider_name, latency, weight_old)

    def get_error_rate(self, provider_name: str) -> float:
        if not self._client:
            return self._fallback.get_error_rate(provider_name)
        try:
            val = self._client.get(f"llm:error_rate:{provider_name}")
            return float(val) if val is not None else 0.0
        except Exception:
            return self._fallback.get_error_rate(provider_name)

    def update_error_rate(self, provider_name: str, is_error: bool, weight_old: float = 0.7) -> float:
        if not self._client:
            return self._fallback.update_error_rate(provider_name, is_error, weight_old)
        try:
            err_val = 1.0 if is_error else 0.0
            current = self.get_error_rate(provider_name)
            new_val = err_val if current is None else current * weight_old + err_val * (1.0 - weight_old)
            self._client.set(f"llm:error_rate:{provider_name}", new_val)
            return new_val
        except Exception:
            return self._fallback.update_error_rate(provider_name, is_error, weight_old)

    def get_cost(self, provider_name: str) -> Optional[float]:
        if not self._client:
            return self._fallback.get_cost(provider_name)
        try:
            val = self._client.get(f"llm:cost:{provider_name}")
            return float(val) if val is not None else None
        except Exception:
            return self._fallback.get_cost(provider_name)

    def set_cost(self, provider_name: str, cost: float) -> None:
        if not self._client:
            self._fallback.set_cost(provider_name, cost)
            return
        try:
            self._client.set(f"llm:cost:{provider_name}", cost)
        except Exception:
            self._fallback.set_cost(provider_name, cost)

    def reset(self) -> None:
        if self._client:
            try:
                keys = self._client.keys("llm:*")
                if keys:
                    self._client.delete(*keys)
            except Exception:
                pass
        self._fallback.reset()


_default_metrics_store: Optional[MetricsStore] = None

def get_metrics_store() -> MetricsStore:
    """Factory retrieving the configured MetricsStore singleton."""
    global _default_metrics_store
    if _default_metrics_store is None:
        if settings.REDIS_URL:
            _default_metrics_store = RedisMetricsStore(settings.REDIS_URL)
        else:
            _default_metrics_store = InMemoryMetricsStore()
    return _default_metrics_store

