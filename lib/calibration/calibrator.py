"""CalibrationData, Calibrator: 過去実績からの補正係数算出."""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DayRecord:
    """1日の実績記録."""

    date: str  # "2026-02-26"
    actual_hours: float  # 手動記録
    location: str  # "office" | "remote" | "unknown"
    notes: str = ""  # "体調不良" etc.


@dataclass
class CalibrationData:
    """キャリブレーションデータ."""

    records: list[DayRecord] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> "CalibrationData":
        """JSONファイルから読み込み."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(records=[DayRecord(**r) for r in data["records"]])

    def save(self, path: Path) -> None:
        """JSONファイルに保存."""
        data = {
            "records": [
                {
                    "date": r.date,
                    "actual_hours": r.actual_hours,
                    "location": r.location,
                    "notes": r.notes,
                }
                for r in self.records
            ]
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


class Calibrator:
    """Active時間の補正を行う."""

    def __init__(self, data: CalibrationData):
        self.data = data

    def compute_coefficients(
        self, daily_active: dict[str, float] | None = None
    ) -> dict[str, float]:
        """条件別の補正係数を算出.

        daily_active: {date_str: active_minutes} のマップ（計算済みのActive時間）
        """
        if daily_active is None:
            return {"office": 1.0, "remote": 1.0, "default": 1.0}

        office_ratios: list[float] = []
        remote_ratios: list[float] = []
        all_ratios: list[float] = []

        for record in self.data.records:
            active = daily_active.get(record.date, 0.0)
            if active <= 0:
                continue
            actual_min = record.actual_hours * 60
            ratio = actual_min / active

            all_ratios.append(ratio)
            if record.location == "office":
                office_ratios.append(ratio)
            elif record.location == "remote":
                remote_ratios.append(ratio)

        def _mean(vals: list[float]) -> float:
            return sum(vals) / len(vals) if vals else 1.0

        return {
            "office": _mean(office_ratios),
            "remote": _mean(remote_ratios),
            "default": _mean(all_ratios),
        }

    def calibrate(
        self,
        active_minutes: float,
        location: str = "unknown",
        daily_active: dict[str, float] | None = None,
    ) -> float:
        """Active時間を補正."""
        coeffs = self.compute_coefficients(daily_active)
        coeff = coeffs.get(location, coeffs["default"])
        return active_minutes * coeff

    def detect_anomalies(
        self, daily_active: dict[str, float], daily_commits: dict[str, int]
    ) -> list[dict]:
        """Active高/commit低 = 体調不良等の異常日を検出."""
        if not daily_active or not daily_commits:
            return []

        # 平均Active時間と平均commit数を計算
        avg_active = sum(daily_active.values()) / max(len(daily_active), 1)
        avg_commits = sum(daily_commits.values()) / max(len(daily_commits), 1)

        anomalies: list[dict] = []
        for date, active in daily_active.items():
            commits = daily_commits.get(date, 0)
            # Active時間が平均以上 かつ commit数が平均の半分以下
            if active >= avg_active and (
                avg_commits == 0 or commits <= avg_commits * 0.5
            ):
                anomalies.append(
                    {
                        "date": date,
                        "active_minutes": active,
                        "commits": commits,
                        "reason": "高Active/低commit",
                    }
                )
        return anomalies
