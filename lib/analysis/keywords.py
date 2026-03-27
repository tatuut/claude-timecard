"""select_task_keywords (玉ねぎ層), keyword分布分析."""

import math
from collections import Counter


def select_task_keywords(
    block_word_counts: list[Counter],
    tfidf: dict[str, float] | None,
    top_n: int = 20,
    min_df_ratio: int = 5,
) -> tuple[list[str], list[str]]:
    """玉ねぎ層アプローチでタスク推定用キーワードを選定.

    Returns:
        task_keywords: タスク推定に使うキーワード（内層）
        context_keywords: 文脈として表示するキーワード（外皮）
    """
    n_blocks = len(block_word_counts)
    word_df: Counter = Counter()
    for wc in block_word_counts:
        for w in wc:
            word_df[w] += 1

    min_df = min(3, max(1, n_blocks // min_df_ratio))

    candidate_n = top_n * 3
    if tfidf:
        scored = []
        for w, s in tfidf.items():
            df = word_df.get(w, 0)
            if df >= min_df:
                scored.append((w, s * math.sqrt(df), df, s))
        scored.sort(key=lambda x: -x[1])
        candidates = scored[:candidate_n]
    else:
        total_c: Counter = Counter()
        for wc in block_word_counts:
            total_c.update(wc)
        candidates = [
            (w, float(c), word_df[w], float(c))
            for w, c in total_c.most_common(candidate_n)
            if word_df[w] >= min_df
        ]

    if not candidates:
        return [], []

    dfs = [df for _, _, df, _ in candidates]
    raw_scores = [raw for _, _, _, raw in candidates]
    median_df = sorted(dfs)[len(dfs) // 2]
    median_raw = sorted(raw_scores)[len(raw_scores) // 2]

    context_kws: list[str] = []
    task_kws: list[str] = []

    for w, score, df, raw in candidates:
        if df > median_df and raw <= median_raw:
            context_kws.append(w)
        else:
            if len(task_kws) < top_n:
                task_kws.append(w)

    return task_kws, context_kws
