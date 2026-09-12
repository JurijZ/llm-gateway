from enum import Enum
from typing import Dict, Optional
import time
import logging
import threading
from app.core.config import settings

logger = logging.getLogger(__name__)

class CircuitState(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"

class CircuitBreakerOpenError(Exception):
    """Raised when an upstream provider or candidate chain is blocked by open circuit breaker."""
    pass

class CircuitBreaker:
    """
    Circuit breaker tracking upstream provider availability.
    Fails fast when a provider experiences repeated failures, preventing
    cascading latency spikes and event-loop blocking.

    Thread-safety: all state mutations are protected by a single threading.Lock,
    making get_state / can_execute / record_success / record_failure safe to call
    from concurrent asyncio coroutines and threads simultaneously.
    """
    def __init__(
        self,
        failure_threshold: Optional[int] = None,
        recovery_timeout: Optional[float] = None,
    ):
        self.failure_threshold = failure_threshold or settings.CIRCUIT_BREAKER_FAILURES
        self.recovery_timeout = recovery_timeout or settings.CIRCUIT_BREAKER_RECOVERY_TIMEOUT
        self.failure_threshold = failure_threshold if failure_threshold is not None else settings.CIRCUIT_BREAKER_FAILURES
        self.recovery_timeout = recovery_timeout if recovery_timeout is not None else settings.CIRCUIT_BREAKER_RECOVERY_TIMEOUT
        self._states: Dict[str, CircuitState] = {}
        self._failure_counts: Dict[str, int] = {}
        self._last_state_change: Dict[str, float] = {}
        self._probe_active: Dict[str, bool] = {}
        # Single lock guarding all mutable state — prevents TOCTOU race between
        # get_state (OPEN → HALF_OPEN transition) and can_execute (_probe_active write).
        self._lock = threading.Lock()

    def get_state(self, provider_name: str) -> CircuitState:
    def _get_state_unlocked(self, provider_name: str) -> CircuitState:
        """Evaluate (and advance) circuit state. MUST be called with self._lock held."""
        now = time.time()
        current_state = self._states.get(provider_name, CircuitState.CLOSED)

        if current_state == CircuitState.OPEN:
            last_change = self._last_state_change.get(provider_name, 0.0)
            if (now - last_change) >= self.recovery_timeout:
                logger.info(f"Circuit for provider '{provider_name}' entered HALF_OPEN state (ready for canary probe).")
                logger.info(
                    f"Circuit for provider '{provider_name}' entered HALF_OPEN state "
                    f"(ready for canary probe)."
                )
                self._states[provider_name] = CircuitState.HALF_OPEN
                self._probe_active[provider_name] = False
                return CircuitState.HALF_OPEN

        return current_state

    def get_state(self, provider_name: str) -> CircuitState:
        with self._lock:
            return self._get_state_unlocked(provider_name)

    def can_execute(self, provider_name: str) -> bool:
        """Returns True if the provider can receive requests; False if the circuit is OPEN."""
        state = self.get_state(provider_name)
        if state == CircuitState.CLOSED:
            return True
        if state == CircuitState.HALF_OPEN:
            # Allow one canary probe request at a time
            if not self._probe_active.get(provider_name, False):
                self._probe_active[provider_name] = True
        """
        Returns True if the provider can receive requests; False if the circuit is OPEN.

        The state evaluation and probe-flag write are performed atomically within a single
        lock acquisition, preventing two concurrent coroutines from both passing the
        HALF_OPEN canary-probe check.
        """
        with self._lock:
            state = self._get_state_unlocked(provider_name)
            if state == CircuitState.CLOSED:
                return True
            if state == CircuitState.HALF_OPEN:
                # Allow exactly one canary probe at a time.
                if not self._probe_active.get(provider_name, False):
                    self._probe_active[provider_name] = True
                    return True
            return False
        return False

    def record_success(self, provider_name: str) -> None:
        """Record successful execution. Closes circuit and resets counters."""
        prev_state = self._states.get(provider_name, CircuitState.CLOSED)
        self._failure_counts[provider_name] = 0
        self._states[provider_name] = CircuitState.CLOSED
        self._probe_active[provider_name] = False
        with self._lock:
            prev_state = self._states.get(provider_name, CircuitState.CLOSED)
            self._failure_counts[provider_name] = 0
            self._states[provider_name] = CircuitState.CLOSED
            self._probe_active[provider_name] = False
        if prev_state != CircuitState.CLOSED:
            logger.info(f"Circuit for provider '{provider_name}' successfully recovered and is now CLOSED.")

    def record_failure(self, provider_name: str) -> None:
        """Record failed execution. Tripping to OPEN when threshold reached."""
        count = self._failure_counts.get(provider_name, 0) + 1
        self._failure_counts[provider_name] = count
        self._probe_active[provider_name] = False
        now = time.time()
        with self._lock:
            count = self._failure_counts.get(provider_name, 0) + 1
            self._failure_counts[provider_name] = count
            self._probe_active[provider_name] = False
            now = time.time()

        state = self._states.get(provider_name, CircuitState.CLOSED)
        if state == CircuitState.HALF_OPEN or count >= self.failure_threshold:
            self._states[provider_name] = CircuitState.OPEN
            self._last_state_change[provider_name] = now
            logger.warning(
                f"Circuit for provider '{provider_name}' tripped to OPEN "
                f"({count} consecutive failures). Bypassing for {self.recovery_timeout}s."
            )
            state = self._states.get(provider_name, CircuitState.CLOSED)
            if state == CircuitState.HALF_OPEN or count >= self.failure_threshold:
                self._states[provider_name] = CircuitState.OPEN
                self._last_state_change[provider_name] = now
                logger.warning(
                    f"Circuit for provider '{provider_name}' tripped to OPEN "
                    f"({count} consecutive failures). Bypassing for {self.recovery_timeout}s."
                )

    def reset(self, provider_name: Optional[str] = None) -> None:
        """Reset state (primarily for tests)."""
        if provider_name:
            self._states.pop(provider_name, None)
            self._failure_counts.pop(provider_name, None)
            self._last_state_change.pop(provider_name, None)
            self._probe_active.pop(provider_name, None)
        else:
            self._states.clear()
            self._failure_counts.clear()
            self._last_state_change.clear()
            self._probe_active.clear()
        with self._lock:
            if provider_name:
                self._states.pop(provider_name, None)
                self._failure_counts.pop(provider_name, None)
                self._last_state_change.pop(provider_name, None)
                self._probe_active.pop(provider_name, None)
            else:
                self._states.clear()
                self._failure_counts.clear()
                self._last_state_change.clear()
                self._probe_active.clear()

_default_circuit_breaker: Optional[CircuitBreaker] = None

def get_circuit_breaker() -> CircuitBreaker:
    global _default_circuit_breaker
    if _default_circuit_breaker is None:
        _default_circuit_breaker = CircuitBreaker()
    return _default_circuit_breaker

