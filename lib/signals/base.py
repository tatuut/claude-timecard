"""SignalProtocol (Protocol class)."""

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from ..parser.events import Event
from ..config import TimecardConfig


@dataclass
class Boundary:
    """検出された境界点."""

    timestamp: datetime
    signal_name: str
    confidence: str  # "hard" | "soft" | "auxiliary"
    reason: str = ""


class Signal(Protocol):
    """境界検出シグナルのProtocol."""

    name: str
    confidence: str  # "hard" | "soft" | "auxiliary"

    def detect(self, events: list[Event], config: TimecardConfig) -> list[Boundary]:
        """境界点を検出."""
        ...
