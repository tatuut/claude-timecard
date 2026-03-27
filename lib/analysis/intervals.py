"""measure_union (sweep line), union-of-intervals射影."""

from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class Interval:
    """時間区間."""

    start: datetime
    end: datetime
    stream: tuple[str, str] | None = None


def measure_union(intervals: list[Interval]) -> float:
    """区間のunionの総分数を計算（sweep line）."""
    if not intervals:
        return 0.0

    endpoints: list[tuple[datetime, int]] = []
    for iv in intervals:
        endpoints.append((iv.start, +1))
        endpoints.append((iv.end, -1))
    endpoints.sort(key=lambda x: x[0])

    active = 0
    total = timedelta()
    prev_time = None
    for time, delta in endpoints:
        if active > 0 and prev_time:
            total += time - prev_time
        active += delta
        prev_time = time
    return total.total_seconds() / 60
