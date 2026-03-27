"""KernelProtocol, GaussianKernel, EpanechnikovKernel, KDE密度計算."""

import math
from collections import Counter
from datetime import datetime, timedelta
from typing import Protocol


class KernelProtocol(Protocol):
    """カーネル関数のProtocol."""

    def evaluate(self, x: float) -> float:
        ...


class GaussianKernel:
    """Gaussianカーネル."""

    def evaluate(self, x: float) -> float:
        return math.exp(-0.5 * x * x)


class EpanechnikovKernel:
    """Epanechnikovカーネル."""

    def evaluate(self, x: float) -> float:
        if abs(x) > 1.0:
            return 0.0
        return 0.75 * (1.0 - x * x)


def get_kernel(name: str = "gaussian") -> KernelProtocol:
    """カーネル名からカーネルオブジェクトを取得."""
    kernels: dict[str, KernelProtocol] = {
        "gaussian": GaussianKernel(),
        "epanechnikov": EpanechnikovKernel(),
    }
    return kernels.get(name, GaussianKernel())


def compute_kde_densities(
    blocks: list,
    block_word_counts: list[Counter],
    keywords: list[str],
    n_samples: int = 200,
    kernel: KernelProtocol | None = None,
) -> tuple[list[datetime], dict[str, list[float]], float]:
    """各キーワードのKDE密度ベクトルを計算.

    Returns:
        sample_times: サンプル時刻のリスト
        densities: {keyword: density_vector}
        sigma_sec: カーネルのσ（秒）
    """
    if kernel is None:
        kernel = GaussianKernel()

    t_min = min(b.start for b in blocks)
    t_max = max(b.end for b in blocks)
    total_span = (t_max - t_min).total_seconds()

    sample_times = [
        t_min + timedelta(seconds=total_span * i / (n_samples - 1))
        for i in range(n_samples)
    ]
    span_days = total_span / 86400
    sigma_sec = total_span * (
        0.03 if span_days <= 2 else 0.025 if span_days <= 7 else 0.015
    )

    densities: dict[str, list[float]] = {}
    for word in keywords:
        d = [0.0] * n_samples
        for bi, b in enumerate(blocks):
            wc = block_word_counts[bi].get(word, 0)
            if wc == 0:
                continue
            w = math.log1p(wc)
            center_sec = ((b.start + (b.end - b.start) / 2) - t_min).total_seconds()
            for si in range(n_samples):
                t_sec = (sample_times[si] - t_min).total_seconds()
                x = (t_sec - center_sec) / sigma_sec
                d[si] += w * kernel.evaluate(x)
        densities[word] = d

    return sample_times, densities, sigma_sec


def find_density_peaks(density: list[float], min_ratio: float = 0.15) -> list[int]:
    """KDE密度のピーク（極大値）インデックスを返す."""
    if not density:
        return []
    max_d = max(density)
    if max_d == 0:
        return []
    thr = max_d * min_ratio
    peaks = []
    for i in range(1, len(density) - 1):
        if (
            density[i] >= thr
            and density[i] >= density[i - 1]
            and density[i] >= density[i + 1]
        ):
            peaks.append(i)
    return peaks
