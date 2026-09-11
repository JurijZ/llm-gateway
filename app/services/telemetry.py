import threading
from typing import Dict, List, Optional
import math

class TelemetryCollector:
    """
    Thread-safe in-memory telemetry collector tracking request counts,
    TTFC percentiles (p50, p95, p99), and fallback trigger frequency.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self.total_requests: int = 0
        self.successful_requests: int = 0
        self.failed_requests: int = 0
        self.fallback_triggers: int = 0
        self.provider_requests: Dict[str, int] = {}
        self.model_requests: Dict[str, int] = {}
        self.ttfc_latencies_ms: List[float] = []
        self.total_durations_ms: List[float] = []

    def record_request_start(self) -> None:
        with self._lock:
            self.total_requests += 1

    def record_fallback(self) -> None:
        with self._lock:
            self.fallback_triggers += 1

    def record_ttfc(self, ttfc_ms: float, provider: Optional[str] = None, model: Optional[str] = None) -> None:
        with self._lock:
            self.ttfc_latencies_ms.append(ttfc_ms)
            if provider:
                self.provider_requests[provider] = self.provider_requests.get(provider, 0) + 1
            if model:
                self.model_requests[model] = self.model_requests.get(model, 0) + 1

    def record_request_complete(
        self,
        duration_ms: float,
        success: bool = True,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        with self._lock:
            if success:
                self.successful_requests += 1
            else:
                self.failed_requests += 1
            self.total_durations_ms.append(duration_ms)

    @staticmethod
    def _calc_percentile(data: List[float], percentile: float) -> Optional[float]:
        if not data:
            return None
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * percentile
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return round(sorted_data[int(k)], 2)
        d0 = sorted_data[int(f)] * (c - k)
        d1 = sorted_data[int(c)] * (k - f)
        return round(d0 + d1, 2)

    def get_metrics(self) -> Dict:
        with self._lock:
            ttfc_p50 = self._calc_percentile(self.ttfc_latencies_ms, 0.50)
            ttfc_p95 = self._calc_percentile(self.ttfc_latencies_ms, 0.95)
            ttfc_p99 = self._calc_percentile(self.ttfc_latencies_ms, 0.99)
            ttfc_avg = (
                round(sum(self.ttfc_latencies_ms) / len(self.ttfc_latencies_ms), 2)
                if self.ttfc_latencies_ms
                else None
            )

            dur_p50 = self._calc_percentile(self.total_durations_ms, 0.50)
            dur_p95 = self._calc_percentile(self.total_durations_ms, 0.95)
            dur_p99 = self._calc_percentile(self.total_durations_ms, 0.99)
            dur_avg = (
                round(sum(self.total_durations_ms) / len(self.total_durations_ms), 2)
                if self.total_durations_ms
                else None
            )

            return {
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "fallback_triggers": self.fallback_triggers,
                "requests_by_provider": dict(self.provider_requests),
                "requests_by_model": dict(self.model_requests),
                "ttfc_ms": {
                    "p50": ttfc_p50,
                    "p95": ttfc_p95,
                    "p99": ttfc_p99,
                    "avg": ttfc_avg,
                    "count": len(self.ttfc_latencies_ms),
                },
                "duration_ms": {
                    "p50": dur_p50,
                    "p95": dur_p95,
                    "p99": dur_p99,
                    "avg": dur_avg,
                    "count": len(self.total_durations_ms),
                },
            }

    def reset(self) -> None:
        with self._lock:
            self.total_requests = 0
            self.successful_requests = 0
            self.failed_requests = 0
            self.fallback_triggers = 0
            self.provider_requests.clear()
            self.model_requests.clear()
            self.ttfc_latencies_ms.clear()
            self.total_durations_ms.clear()

_telemetry_singleton: Optional[TelemetryCollector] = None

def get_telemetry() -> TelemetryCollector:
    global _telemetry_singleton
    if _telemetry_singleton is None:
        _telemetry_singleton = TelemetryCollector()
    return _telemetry_singleton

