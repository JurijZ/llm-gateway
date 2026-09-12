import threading
from typing import Dict, List, Optional
from typing import Dict, List, Optional, NamedTuple
from collections import deque
import math
import time

class FallbackEvent(NamedTuple):
    timestamp: float
    failed_provider: str
    reason: str
    next_provider: str


class TelemetryCollector:
    """
    Thread-safe in-memory telemetry collector tracking request counts,
    TTFC percentiles (p50, p95, p99), and fallback trigger frequency.

    §4.1: ttfc_latencies_ms and total_durations_ms are bounded deques (maxlen=10_000)
    to prevent unbounded memory growth under production load.

    §4.3: Fallback events are stored in a bounded ring-buffer with structured context.
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
        # §4.1: Use bounded deques — maxlen=10_000 caps memory at ~80 KB per list
        # and avoids O(n log n) sort cost growing unboundedly with request volume.
        self.ttfc_latencies_ms: deque[float] = deque(maxlen=10_000)
        self.total_durations_ms: deque[float] = deque(maxlen=10_000)
        # §4.3: Structured fallback event ring-buffer (last 200 events).
        self._fallback_events: deque[FallbackEvent] = deque(maxlen=200)

    def record_request_start(self) -> None:
        with self._lock:
            self.total_requests += 1

    def record_fallback(self) -> None:
    def record_fallback(
        self,
        failed_provider: str = "unknown",
        reason: str = "unknown",
        next_provider: str = "unknown",
    ) -> None:
        """
        §4.3: Record a structured fallback event with context.
        Increments the bare counter for backward-compatibility and appends
        a FallbackEvent to the ring-buffer for structured inspection.
        """
        with self._lock:
            self.fallback_triggers += 1
            self._fallback_events.append(
                FallbackEvent(
                    timestamp=time.time(),
                    failed_provider=failed_provider,
                    reason=reason,
                    next_provider=next_provider,
                )
            )

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
    def _calc_percentile(sorted_data: List[float], percentile: float) -> Optional[float]:
        """
        §5.2: Expects a pre-sorted list. Callers must sort once and reuse the sorted
        list for all percentile calculations, rather than re-sorting on every call.
        """
        if not sorted_data:
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
            # §5.2: Sort each list exactly once, then pass the sorted result to all
            # three percentile calls. Previously _calc_percentile sorted internally,
            # causing 6 full sorts per get_metrics() call while holding _lock.
            sorted_ttfc = sorted(self.ttfc_latencies_ms)
            ttfc_p50 = self._calc_percentile(sorted_ttfc, 0.50)
            ttfc_p95 = self._calc_percentile(sorted_ttfc, 0.95)
            ttfc_p99 = self._calc_percentile(sorted_ttfc, 0.99)
            ttfc_avg = (
                round(sum(self.ttfc_latencies_ms) / len(self.ttfc_latencies_ms), 2)
                if self.ttfc_latencies_ms
                round(sum(sorted_ttfc) / len(sorted_ttfc), 2)
                if sorted_ttfc
                else None
            )

            dur_p50 = self._calc_percentile(self.total_durations_ms, 0.50)
            dur_p95 = self._calc_percentile(self.total_durations_ms, 0.95)
            dur_p99 = self._calc_percentile(self.total_durations_ms, 0.99)
            sorted_dur = sorted(self.total_durations_ms)
            dur_p50 = self._calc_percentile(sorted_dur, 0.50)
            dur_p95 = self._calc_percentile(sorted_dur, 0.95)
            dur_p99 = self._calc_percentile(sorted_dur, 0.99)
            dur_avg = (
                round(sum(self.total_durations_ms) / len(self.total_durations_ms), 2)
                if self.total_durations_ms
                round(sum(sorted_dur) / len(sorted_dur), 2)
                if sorted_dur
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
                    "count": len(sorted_ttfc),
                },
                "duration_ms": {
                    "p50": dur_p50,
                    "p95": dur_p95,
                    "p99": dur_p99,
                    "avg": dur_avg,
                    "count": len(self.total_durations_ms),
                    "count": len(sorted_dur),
                },
            }

    def get_fallback_events(self) -> List[dict]:
        """
        §4.3: Return the most recent fallback events (up to 200) as serializable dicts.
        Useful for diagnosing fallback storms in production.
        """
        with self._lock:
            return [
                {
                    "timestamp": ev.timestamp,
                    "failed_provider": ev.failed_provider,
                    "reason": ev.reason,
                    "next_provider": ev.next_provider,
                }
                for ev in self._fallback_events
            ]

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
            self._fallback_events.clear()

_telemetry_singleton: Optional[TelemetryCollector] = None

def get_telemetry() -> TelemetryCollector:
    global _telemetry_singleton
    if _telemetry_singleton is None:
        _telemetry_singleton = TelemetryCollector()
    return _telemetry_singleton

