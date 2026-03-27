"""処理時間計測ユーティリティ."""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class TimingResult:
    """個別計測結果."""

    label: str
    seconds: float


class Timer:
    """処理時間を計測・記録するユーティリティ."""

    def __init__(self):
        self.results: list[TimingResult] = []

    @contextmanager
    def measure(self, label: str):
        t0 = time.perf_counter()
        yield
        dt = time.perf_counter() - t0
        self.results.append(TimingResult(label=label, seconds=dt))

    def summary(self) -> dict:
        total = sum(r.seconds for r in self.results)
        return {
            "total_seconds": round(total, 3),
            "steps": [
                {"label": r.label, "seconds": round(r.seconds, 3)}
                for r in self.results
            ],
        }


# グローバルタイマー
_global_timer: Timer | None = None


def get_timer() -> Timer | None:
    """グローバルタイマーを取得。未有効時はNone."""
    return _global_timer


def enable_timing() -> Timer:
    """グローバルタイマーを有効化して返す."""
    global _global_timer
    _global_timer = Timer()
    return _global_timer


def disable_timing() -> None:
    """グローバルタイマーを無効化."""
    global _global_timer
    _global_timer = None
