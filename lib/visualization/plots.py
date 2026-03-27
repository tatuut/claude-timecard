"""plot_keyword_graph, plot_task_timeline."""

import math
import sys
from collections import Counter
from datetime import timedelta
from pathlib import Path

from ..config import C_GREEN, C_RESET
from ..analysis.blocks import Block
from ..analysis.tfidf import _tokenize


def plot_keyword_graph(
    blocks: list[Block],
    tfidf: dict[str, float] | None = None,
    top_n: int = 10,
    output: str = "timecard_graph.png",
):
    """キーワード時間分布をKDE風の重なり曲線グラフで出力."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.dates import DateFormatter, HourLocator, DayLocator
    except ImportError:
        print("Error: matplotlib が必要です: pip install matplotlib", file=sys.stderr)
        return

    jp_fonts = ["Yu Gothic", "Meiryo", "MS Gothic", "Hiragino Sans"]
    plt.rcParams["font.family"] = ["sans-serif"]
    plt.rcParams["font.sans-serif"] = jp_fonts + plt.rcParams["font.sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False

    if not blocks:
        return

    block_word_counts: list[Counter] = []
    for b in blocks:
        wc: Counter = Counter()
        for msg in b.messages:
            for token in _tokenize(msg):
                wc[token] += 1
        block_word_counts.append(wc)

    word_df: Counter = Counter()
    for wc in block_word_counts:
        for w in wc:
            word_df[w] += 1

    min_df = min(3, max(1, len(blocks) // 5))
    if tfidf:
        scored = []
        for w, s in tfidf.items():
            df = word_df.get(w, 0)
            if df >= min_df:
                scored.append((w, s * math.sqrt(df)))
        scored.sort(key=lambda x: -x[1])
        display_words = [w for w, _ in scored[:top_n]]
    else:
        total_counts: Counter = Counter()
        for wc in block_word_counts:
            total_counts.update(wc)
        candidates = [
            (w, c) for w, c in total_counts.most_common(50) if word_df[w] >= min_df
        ]
        display_words = [w for w, _ in candidates[:top_n]]

    if not display_words:
        return

    t_min = min(b.start for b in blocks)
    t_max = max(b.end for b in blocks)
    total_span = (t_max - t_min).total_seconds()
    if total_span == 0:
        return

    n_samples = 500
    sample_times = [
        t_min + timedelta(seconds=total_span * i / (n_samples - 1))
        for i in range(n_samples)
    ]

    span_days = total_span / 86400
    if span_days <= 2:
        sigma_sec = total_span * 0.03
    elif span_days <= 7:
        sigma_sec = total_span * 0.025
    else:
        sigma_sec = total_span * 0.015

    word_densities: dict[str, list[float]] = {}
    for word in display_words:
        density = [0.0] * n_samples
        for bi, b in enumerate(blocks):
            wc = block_word_counts[bi].get(word, 0)
            if wc == 0:
                continue
            weight = math.log1p(wc)
            center = b.start + (b.end - b.start) / 2
            center_sec = (center - t_min).total_seconds()
            for si in range(n_samples):
                t_sec = (sample_times[si] - t_min).total_seconds()
                dist = (t_sec - center_sec) / sigma_sec
                density[si] += weight * math.exp(-0.5 * dist * dist)
        word_densities[word] = density

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    colors = [
        "#e94560", "#0f3460", "#53d8fb", "#f9a825", "#ab47bc",
        "#66bb6a", "#ef5350", "#42a5f5", "#ff7043", "#26c6da",
        "#ec407a", "#7e57c2", "#9ccc65", "#ffa726", "#29b6f6",
    ]

    for i, word in enumerate(display_words):
        density = word_densities[word]
        max_d = max(density) if max(density) > 0 else 1.0
        normalized = [d / max_d for d in density]
        color = colors[i % len(colors)]
        ax.fill_between(
            sample_times, normalized,
            alpha=0.25, color=color, linewidth=0,
        )
        ax.plot(
            sample_times, normalized,
            color=color, linewidth=1.5, alpha=0.8, label=word,
        )

    ax.set_xlim(t_min, t_max)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("相対密度", color="#aaa", fontsize=10)
    ax.set_title(
        f"キーワード時間分布  {t_min.strftime('%m/%d')}〜{t_max.strftime('%m/%d')}",
        color="white", fontsize=13, pad=12,
    )

    if span_days > 3:
        ax.xaxis.set_major_locator(DayLocator())
        ax.xaxis.set_major_formatter(DateFormatter("%m/%d"))
    else:
        ax.xaxis.set_major_locator(HourLocator(interval=3))
        ax.xaxis.set_major_formatter(DateFormatter("%m/%d %H:%M"))

    ax.tick_params(colors="#888", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#333")
    ax.grid(axis="x", color="#333", alpha=0.5, linestyle="--")

    ax.legend(
        loc="upper right", fontsize=10, ncol=2,
        facecolor="#1a1a2e", edgecolor="#444", labelcolor="white",
        framealpha=0.9, borderpad=1,
    )

    plt.tight_layout()
    plt.savefig(output, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    abs_path = str(Path(output).resolve())
    print(f"\n  {C_GREEN}グラフ保存: {abs_path}{C_RESET}")


def plot_keyword_ranking(
    blocks: list[Block],
    tfidf: dict[str, float] | None = None,
    top_n: int = 20,
    output: str = "timecard_keywords.png",
):
    """キーワードTF-IDFランキングの水平バーチャートを出力.

    unigramとbigramが混在してどちらが上位かが一目でわかる。
    bigram（_を含む）は別色で表示。
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Error: matplotlib が必要です: pip install matplotlib", file=sys.stderr)
        return

    jp_fonts = ["Yu Gothic", "Meiryo", "MS Gothic", "Hiragino Sans"]
    plt.rcParams["font.family"] = ["sans-serif"]
    plt.rcParams["font.sans-serif"] = jp_fonts + plt.rcParams["font.sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False

    if not blocks or not tfidf:
        return

    # ブロック語彙
    block_word_counts: list[Counter] = []
    for b in blocks:
        wc: Counter = Counter()
        for msg in b.messages:
            for token in _tokenize(msg):
                wc[token] += 1
        block_word_counts.append(wc)

    word_df: Counter = Counter()
    for wc in block_word_counts:
        for w in wc:
            word_df[w] += 1

    total_tf: Counter = Counter()
    for wc in block_word_counts:
        total_tf.update(wc)

    min_df = min(3, max(1, len(blocks) // 5))
    scored = []
    for w, s in tfidf.items():
        df = word_df.get(w, 0)
        if df >= min_df:
            scored.append((w, s * math.sqrt(df), total_tf.get(w, 0), df))
    scored.sort(key=lambda x: -x[1])
    top = scored[:top_n]

    if not top:
        return

    # 描画
    words = [t[0] for t in reversed(top)]
    scores = [t[1] for t in reversed(top)]
    counts = [t[2] for t in reversed(top)]
    is_bigram = ["_" in w for w in words]

    fig_h = max(4, 0.35 * len(words) + 1.5)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, fig_h), width_ratios=[3, 1])
    fig.patch.set_facecolor("#1a1a2e")

    # 左: TF-IDF スコアバー
    ax1.set_facecolor("#16213e")
    bar_colors = ["#53d8fb" if bg else "#e94560" for bg in is_bigram]
    ax1.barh(range(len(words)), scores, color=bar_colors, alpha=0.85, height=0.7)
    ax1.set_yticks(range(len(words)))
    ax1.set_yticklabels(words, color="white", fontsize=11)
    ax1.set_xlabel("TF-IDF × √DF", color="#aaa", fontsize=10)
    ax1.set_title("キーワードランキング", color="white", fontsize=13, pad=12)
    ax1.tick_params(colors="#888", labelsize=9)
    for spine in ax1.spines.values():
        spine.set_color("#333")
    ax1.grid(axis="x", color="#333", alpha=0.4, linestyle="--")

    # 凡例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#e94560", alpha=0.85, label="unigram"),
        Patch(facecolor="#53d8fb", alpha=0.85, label="bigram"),
    ]
    ax1.legend(
        handles=legend_elements, loc="lower right", fontsize=10,
        facecolor="#1a1a2e", edgecolor="#444", labelcolor="white",
    )

    # 右: 出現回数バー
    ax2.set_facecolor("#16213e")
    ax2.barh(range(len(words)), counts, color=bar_colors, alpha=0.5, height=0.7)
    ax2.set_yticks([])
    ax2.set_xlabel("出現回数", color="#aaa", fontsize=10)
    ax2.set_title("出現数", color="white", fontsize=13, pad=12)
    ax2.tick_params(colors="#888", labelsize=9)
    for spine in ax2.spines.values():
        spine.set_color("#333")
    ax2.grid(axis="x", color="#333", alpha=0.4, linestyle="--")

    plt.tight_layout()
    plt.savefig(output, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    abs_path = str(Path(output).resolve())
    print(f"\n  {C_GREEN}キーワードランキング保存: {abs_path}{C_RESET}")


def plot_task_timeline(
    tasks: list,
    blocks: list[Block],
    context_keywords: list[str] | None = None,
    output: str = "timecard_tasks.png",
):
    """タスクごとのブロックをガントチャート風にタイムライン表示."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.dates import DateFormatter, DayLocator, HourLocator
    except ImportError:
        print("Error: matplotlib が必要です: pip install matplotlib", file=sys.stderr)
        return

    jp_fonts = ["Yu Gothic", "Meiryo", "MS Gothic", "Hiragino Sans"]
    plt.rcParams["font.family"] = ["sans-serif"]
    plt.rcParams["font.sans-serif"] = jp_fonts + plt.rcParams["font.sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False

    if not tasks or not blocks:
        return

    rows: list[tuple[str, str, set[int], str]] = []
    colors = [
        "#e94560", "#53d8fb", "#f9a825", "#66bb6a", "#ab47bc",
        "#42a5f5", "#ff7043", "#26c6da", "#ec407a", "#7e57c2",
    ]
    for ti, task in enumerate(tasks):
        c = colors[ti % len(colors)]
        label = ", ".join(task.keywords[:3])
        rows.append((label, "main", task.block_set, c))
        for child in task.children:
            clabel = "  └ " + ", ".join(child.keywords[:2])
            rel_tag = "含" if child.relationship == "subtask" else "部"
            rows.append((clabel, rel_tag, child.block_set, c + "88"))

    if not rows:
        return

    n_rows = len(rows)
    fig_h = max(3, 0.5 * n_rows + 1.5)
    fig, ax = plt.subplots(figsize=(14, fig_h))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    t_min = min(b.start for b in blocks)
    t_max = max(b.end for b in blocks)
    span_days = (t_max - t_min).total_seconds() / 86400

    bar_height = 0.6
    for ri, (label, rel, bset, color) in enumerate(rows):
        y = n_rows - 1 - ri
        for bi in bset:
            if bi >= len(blocks):
                continue
            b = blocks[bi]
            ax.barh(
                y,
                (b.end - b.start).total_seconds() / 86400,
                left=b.start,
                height=bar_height,
                color=color,
                alpha=0.7 if rel == "main" else 0.45,
                edgecolor="none",
            )
        ax.text(
            t_min - timedelta(seconds=(t_max - t_min).total_seconds() * 0.01),
            y,
            label,
            ha="right",
            va="center",
            color="white" if rel == "main" else "#aaa",
            fontsize=8,
            fontweight="bold" if rel == "main" else "normal",
        )

    ax.set_ylim(-0.5, n_rows - 0.5)
    ax.set_xlim(t_min, t_max)
    ax.set_yticks([])

    if span_days > 3:
        ax.xaxis.set_major_locator(DayLocator())
        ax.xaxis.set_major_formatter(DateFormatter("%m/%d"))
    else:
        ax.xaxis.set_major_locator(HourLocator(interval=3))
        ax.xaxis.set_major_formatter(DateFormatter("%m/%d %H:%M"))

    ax.tick_params(colors="#888", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#333")
    ax.grid(axis="x", color="#333", alpha=0.4, linestyle="--")

    title = f"タスクタイムライン  {t_min.strftime('%m/%d')}〜{t_max.strftime('%m/%d')}"
    ax.set_title(title, color="white", fontsize=13, pad=12)

    if context_keywords:
        ctx_str = "文脈: " + ", ".join(context_keywords[:6])
        ax.text(
            0.01, -0.08, ctx_str,
            transform=ax.transAxes, color="#888", fontsize=7,
        )

    plt.tight_layout()
    plt.savefig(output, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    abs_path = str(Path(output).resolve())
    print(f"\n  {C_GREEN}タイムライン保存: {abs_path}{C_RESET}")


def plot_branch_timeline(
    blocks: list[Block],
    output: str = "timecard_branches.png",
):
    """ブランチごとのガントチャートを描画.

    各ブランチを1行として、いつ作業していたかを表示する。
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.dates import DateFormatter, DayLocator, HourLocator
    except ImportError:
        print("Error: matplotlib が必要です: pip install matplotlib", file=sys.stderr)
        return

    jp_fonts = ["Yu Gothic", "Meiryo", "MS Gothic", "Hiragino Sans"]
    plt.rcParams["font.family"] = ["sans-serif"]
    plt.rcParams["font.sans-serif"] = jp_fonts + plt.rcParams["font.sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False

    if not blocks:
        return

    # ブランチごとにブロックを集約
    branch_blocks: dict[str, list[Block]] = {}
    for b in blocks:
        # 代表ブランチを取得
        meaningful = [
            br for br in b.branches
            if br and br not in ("main", "master", "HEAD", "development")
        ]
        if meaningful:
            from collections import Counter as _Counter
            top_branch = _Counter(meaningful).most_common(1)[0][0]
        else:
            top_branch = "main"

        if top_branch not in branch_blocks:
            branch_blocks[top_branch] = []
        branch_blocks[top_branch].append(b)

    if not branch_blocks:
        return

    # Active時間でソート
    branch_order = sorted(
        branch_blocks.keys(),
        key=lambda br: sum(
            (b.end - b.start).total_seconds() for b in branch_blocks[br]
        ),
        reverse=True,
    )

    # 上位20ブランチに制限
    branch_order = branch_order[:20]

    # ブランチ名を短縮
    def _short_branch(name: str) -> str:
        parts = name.split("/")
        if len(parts) >= 3:
            return "/".join(parts[2:])
        if len(parts) == 2:
            return parts[1]
        return name

    n_rows = len(branch_order)
    fig_h = max(3, 0.4 * n_rows + 1.5)
    fig, ax = plt.subplots(figsize=(14, fig_h))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    t_min = min(b.start for b in blocks)
    t_max = max(b.end for b in blocks)
    span_days = (t_max - t_min).total_seconds() / 86400

    colors = [
        "#e94560", "#53d8fb", "#f9a825", "#66bb6a", "#ab47bc",
        "#42a5f5", "#ff7043", "#26c6da", "#ec407a", "#7e57c2",
        "#9ccc65", "#ffa726", "#29b6f6", "#ef5350", "#0f3460",
    ]

    bar_height = 0.6
    for ri, branch in enumerate(branch_order):
        y = n_rows - 1 - ri
        color = colors[ri % len(colors)]
        for b in branch_blocks[branch]:
            ax.barh(
                y,
                (b.end - b.start).total_seconds() / 86400,
                left=b.start,
                height=bar_height,
                color=color,
                alpha=0.7,
                edgecolor="none",
            )
        short = _short_branch(branch)
        ax.text(
            t_min - timedelta(seconds=(t_max - t_min).total_seconds() * 0.01),
            y,
            short,
            ha="right",
            va="center",
            color="white",
            fontsize=8,
        )

    ax.set_ylim(-0.5, n_rows - 0.5)
    ax.set_xlim(t_min, t_max)
    ax.set_yticks([])

    if span_days > 3:
        ax.xaxis.set_major_locator(DayLocator())
        ax.xaxis.set_major_formatter(DateFormatter("%m/%d"))
    else:
        ax.xaxis.set_major_locator(HourLocator(interval=3))
        ax.xaxis.set_major_formatter(DateFormatter("%m/%d %H:%M"))

    ax.tick_params(colors="#888", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#333")
    ax.grid(axis="x", color="#333", alpha=0.4, linestyle="--")

    title = f"ブランチタイムライン  {t_min.strftime('%m/%d')}〜{t_max.strftime('%m/%d')}"
    ax.set_title(title, color="white", fontsize=13, pad=12)

    plt.tight_layout()
    plt.savefig(output, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    abs_path = str(Path(output).resolve())
    print(f"\n  {C_GREEN}ブランチタイムライン保存: {abs_path}{C_RESET}")
