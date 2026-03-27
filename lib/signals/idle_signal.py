"""idle gap シグナル（ソフト境界）."""

from ..parser.events import Event
from ..config import TimecardConfig
from .base import Boundary
from .registry import register_signal


@register_signal("idle")
class IdleSignal:
    """アイドルギャップをソフト境界として検出."""

    name = "idle"
    confidence = "soft"

    def detect(self, events: list[Event], config: TimecardConfig) -> list[Boundary]:
        """idle_threshold_min を超えるギャップを検出."""
        boundaries: list[Boundary] = []
        for i in range(1, len(events)):
            gap = (events[i].ts - events[i - 1].ts).total_seconds() / 60
            if gap > config.idle_threshold_min:
                boundaries.append(
                    Boundary(
                        timestamp=events[i].ts,
                        signal_name=self.name,
                        confidence=self.confidence,
                        reason=f"idle {gap:.0f}min",
                    )
                )
        return boundaries
