"""TaskNode, infer_task_hierarchy, infer_tasks_branch_first, task_block_set."""

import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime

from .tfidf import _tokenize, build_tfidf
from .kde import compute_kde_densities, find_density_peaks
from .keywords import select_task_keywords
from .embeddings import cosine_sim, overlap_coeff, merge_synonyms_by_embedding
from .streams import group_by_stream
from ..analysis.blocks import Block, build_blocks


@dataclass
class TaskNode:
    """推定タスクノード."""

    keywords: list[str]
    children: list["TaskNode"] = field(default_factory=list)
    active_minutes: float = 0.0
    sigma_minutes: float = 0.0
    peak_times: list[str] = field(default_factory=list)
    relationship: str = "main"  # "main" | "subtask" | "partial"
    block_set: set[int] = field(default_factory=set)
    branches: list[str] = field(default_factory=list)


def task_block_set(
    task_keywords: list[str],
    blocks: list[Block],
    block_word_counts: list[Counter],
    min_hits: int = 2,
) -> tuple[set[int], set[int], float, float]:
    """タスクキーワードが出現するブロック集合を返す.

    Returns:
        certain: 確実に帰属するブロックインデックス集合
        marginal: 境界的なブロック集合
        active_min: 確実ブロックの合計時間(分)
        sigma_min: 境界ブロックによる不確実性σ(分)
    """
    certain: set[int] = set()
    marginal: set[int] = set()

    for bi, wc in enumerate(block_word_counts):
        hits = sum(1 for k in task_keywords if wc.get(k, 0) > 0)
        if hits >= min_hits:
            certain.add(bi)
        elif hits == 1:
            total_count = sum(wc.get(k, 0) for k in task_keywords)
            if total_count >= 3:
                certain.add(bi)
            elif total_count >= 1:
                marginal.add(bi)

    active = sum(
        (blocks[bi].end - blocks[bi].start).total_seconds() / 60 for bi in certain
    )
    marginal_var = sum(
        ((blocks[bi].end - blocks[bi].start).total_seconds() / 60) ** 2 * 0.25
        for bi in marginal
    )
    sigma = math.sqrt(marginal_var) if marginal_var > 0 else 0.0

    return certain, marginal, active, sigma


def infer_task_hierarchy(
    blocks: list[Block],
    tfidf: dict[str, float] | None,
    top_n: int = 20,
) -> tuple[list[TaskNode], list[str]]:
    """KDE分布の重なりからタスク階層を推定."""
    if len(blocks) < 2:
        return [], []

    # ブロック語彙
    block_word_counts: list[Counter] = []
    for b in blocks:
        wc: Counter = Counter()
        for msg in b.messages:
            for token in _tokenize(msg):
                wc[token] += 1
        block_word_counts.append(wc)

    # 玉ねぎ層キーワード選定
    keywords, context_kws = select_task_keywords(block_word_counts, tfidf, top_n)
    if len(keywords) < 2:
        return [], context_kws

    # embedding同義語マージ
    keywords = merge_synonyms_by_embedding(keywords, block_word_counts)

    # KDE密度
    sample_times, densities, _ = compute_kde_densities(
        blocks, block_word_counts, keywords
    )
    n_samples = len(sample_times)

    # --- Phase 1: コサイン類似度行列 ---
    n = len(keywords)
    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        sim[i][i] = 1.0
        for j in range(i + 1, n):
            s = cosine_sim(densities[keywords[i]], densities[keywords[j]])
            sim[i][j] = sim[j][i] = s

    # --- Phase 2: 凝集型クラスタリング（平均リンケージ）---
    clusters: list[list[int]] = [[i] for i in range(n)]
    merge_thr = 0.55
    max_cluster = 5

    while len(clusters) > 1:
        best_s, bi_, bj_ = -1.0, 0, 1
        for ci in range(len(clusters)):
            for cj in range(ci + 1, len(clusters)):
                if len(clusters[ci]) + len(clusters[cj]) > max_cluster:
                    continue
                avg = sum(
                    sim[a][b] for a in clusters[ci] for b in clusters[cj]
                ) / (len(clusters[ci]) * len(clusters[cj]))
                if avg > best_s:
                    best_s, bi_, bj_ = avg, ci, cj
        if best_s < merge_thr:
            break
        clusters[bi_] = clusters[bi_] + clusters[bj_]
        del clusters[bj_]

    # 全キーワード出現数（メイン判定用）
    total_tf: Counter = Counter()
    for wc in block_word_counts:
        total_tf.update(wc)

    # --- Phase 3: 各クラスタ → メイン/サブ分類 + ブロック集合ベースの時間推定 ---
    results: list[TaskNode] = []

    for cidxs in clusters:
        ckws = sorted(
            [keywords[i] for i in cidxs], key=lambda w: -total_tf.get(w, 0)
        )

        if len(ckws) == 1:
            kw = ckws[0]
            cert, marg, act, sig = task_block_set(
                [kw], blocks, block_word_counts, min_hits=1
            )
            pks = find_density_peaks(densities[kw])
            results.append(
                TaskNode(
                    keywords=[kw],
                    active_minutes=act,
                    sigma_minutes=sig,
                    peak_times=[
                        sample_times[p].strftime("%m/%d %H:%M") for p in pks[:5]
                    ],
                    block_set=cert | marg,
                )
            )
            continue

        # 複数キーワード → メイン/サブ/部分を判定
        main_kw = ckws[0]
        main_d = densities[main_kw]
        main_peak = max(main_d) if max(main_d) > 0 else 1.0

        mains = [main_kw]
        subs: list[tuple[str, str]] = []

        for kw in ckws[1:]:
            kw_d = densities[kw]
            contain = overlap_coeff(main_d, kw_d)
            peak_r = (max(kw_d) / main_peak) if main_peak > 0 else 0

            if contain > 0.7 and peak_r > 0.5:
                mains.append(kw)
            elif contain > 0.55:
                subs.append((kw, "subtask"))
            elif contain > 0.25:
                subs.append((kw, "partial"))
            else:
                mains.append(kw)

        # メインタスクのブロック集合
        main_cert, main_marg, main_act, main_sig = task_block_set(
            mains, blocks, block_word_counts
        )
        main_blocks = main_cert | main_marg

        # 合成密度でピーク検出
        composite = [0.0] * n_samples
        for kw in mains:
            for si in range(n_samples):
                composite[si] += densities[kw][si]
        pks = find_density_peaks(composite)

        # 子ノード
        children: list[TaskNode] = []
        for sub_kw, rel in subs:
            sub_cert, sub_marg, sub_act, sub_sig = task_block_set(
                [sub_kw], blocks, block_word_counts, min_hits=1
            )
            sub_blocks = sub_cert | sub_marg

            overlap = sub_blocks & main_blocks
            if len(sub_blocks) > 0:
                overlap_ratio = len(overlap) / len(sub_blocks)
            else:
                overlap_ratio = 0

            if (
                overlap_ratio < 0.3
                or len(sub_blocks) > len(main_blocks)
                or sub_act > main_act
            ):
                sp = find_density_peaks(densities[sub_kw])
                results.append(
                    TaskNode(
                        keywords=[sub_kw],
                        active_minutes=sub_act,
                        sigma_minutes=sub_sig,
                        peak_times=[
                            sample_times[p].strftime("%m/%d %H:%M") for p in sp[:3]
                        ],
                        block_set=sub_blocks,
                    )
                )
            else:
                actual_rel = "subtask" if overlap_ratio > 0.7 else "partial"
                sp = find_density_peaks(densities[sub_kw])
                children.append(
                    TaskNode(
                        keywords=[sub_kw],
                        active_minutes=sub_act,
                        sigma_minutes=sub_sig,
                        peak_times=[
                            sample_times[p].strftime("%m/%d %H:%M") for p in sp[:3]
                        ],
                        relationship=actual_rel,
                        block_set=sub_blocks,
                    )
                )

        children.sort(key=lambda t: -t.active_minutes)

        results.append(
            TaskNode(
                keywords=mains,
                children=children,
                active_minutes=main_act,
                sigma_minutes=main_sig,
                peak_times=[
                    sample_times[p].strftime("%m/%d %H:%M") for p in pks[:5]
                ],
                block_set=main_blocks,
            )
        )

    results.sort(key=lambda t: -t.active_minutes)
    return results, context_kws


def _stream_tfidf_keywords(
    blocks: list[Block],
    tfidf: dict[str, float],
    top_n: int = 8,
) -> list[str]:
    """ストリーム内ブロックのTF-IDFスコア上位キーワードを返す."""
    word_counts: Counter = Counter()
    for b in blocks:
        for msg in b.messages:
            for token in _tokenize(msg):
                word_counts[token] += 1
    scored = []
    for w, c in word_counts.items():
        s = tfidf.get(w, 0.0)
        if s > 0:
            scored.append((w, s * math.log1p(c)))
    scored.sort(key=lambda x: -x[1])
    return [w for w, _ in scored[:top_n]]


def _stream_keyword_vector(
    blocks: list[Block],
    tfidf: dict[str, float],
    vocabulary: list[str],
) -> list[float]:
    """ストリームのブロック群からTF-IDFベースのベクトルを作成."""
    word_counts: Counter = Counter()
    for b in blocks:
        for msg in b.messages:
            for token in _tokenize(msg):
                word_counts[token] += 1
    vec = []
    for w in vocabulary:
        tf = word_counts.get(w, 0)
        score = tfidf.get(w, 0.0)
        vec.append(tf * score)
    return vec


def infer_tasks_branch_first(
    events: list,
    blocks: list[Block],
    tfidf: dict[str, float] | None = None,
    idle_threshold: int = 20,
    per_turn_time: int = 5,
    top_n: int = 20,
    merge_threshold: float = 0.6,
) -> tuple[list[TaskNode], list[str]]:
    """ブランチを第一軸としたタスク推定.

    1. (project, branch) でイベントをグループ化
    2. 各ストリームごとにブロック構築
    3. 各ストリーム内でTF-IDFキーワード抽出
    4. ストリーム間のキーワード類似度でマージ
    5. TaskNode のリストを返す（branches フィールド付き）
    """
    from ..parser.events import Event

    if not blocks:
        return [], []

    # TF-IDFがなければ構築
    if tfidf is None:
        tfidf = build_tfidf(blocks)

    # ストリーム分割
    streams = group_by_stream(events)

    # 各ストリームのブロック構築
    stream_blocks: dict[tuple[str, str], list[Block]] = {}
    for key, stream_events in streams.items():
        sblocks = build_blocks(stream_events, idle_threshold, per_turn_time)
        if sblocks:
            stream_blocks[key] = sblocks

    if not stream_blocks:
        return [], []

    # 全ブロックのインデックスマップ（元のblocks listとの対応）
    block_time_index: dict[str, int] = {}
    for bi, b in enumerate(blocks):
        key = f"{b.start.isoformat()}_{b.end.isoformat()}"
        block_time_index[key] = bi

    # 各ストリームのキーワードと対応ブロックインデックス
    stream_keys = list(stream_blocks.keys())
    stream_kws: dict[tuple[str, str], list[str]] = {}
    stream_block_indices: dict[tuple[str, str], set[int]] = {}

    for skey, sblocks in stream_blocks.items():
        kws = _stream_tfidf_keywords(sblocks, tfidf, top_n=8)
        stream_kws[skey] = kws

        # ストリームブロックを元のブロックリストにマッピング
        indices: set[int] = set()
        for sb in sblocks:
            # 時間範囲の重なりで対応するブロックを探す
            for bi, b in enumerate(blocks):
                if (sb.start <= b.end and sb.end >= b.start):
                    overlap_start = max(sb.start, b.start)
                    overlap_end = min(sb.end, b.end)
                    overlap_sec = (overlap_end - overlap_start).total_seconds()
                    if overlap_sec > 0:
                        indices.add(bi)
        stream_block_indices[skey] = indices

    # 全ストリームのキーワードを統合して語彙を作成
    all_vocab: set[str] = set()
    for kws in stream_kws.values():
        all_vocab.update(kws)
    vocabulary = sorted(all_vocab)

    # コンテキストキーワード: 過半数のストリームに出現する語
    n_streams = len(stream_keys)
    word_stream_count: Counter = Counter()
    for kws in stream_kws.values():
        for w in set(kws):
            word_stream_count[w] += 1
    context_kws = [
        w for w, c in word_stream_count.most_common()
        if c > n_streams * 0.5 and n_streams >= 2
    ][:8]

    # 各ストリームのベクトルを計算
    stream_vectors: dict[tuple[str, str], list[float]] = {}
    for skey in stream_keys:
        vec = _stream_keyword_vector(stream_blocks[skey], tfidf, vocabulary)
        stream_vectors[skey] = vec

    # ストリーム間コサイン類似度でマージ
    clusters: list[list[tuple[str, str]]] = [[k] for k in stream_keys]

    while len(clusters) > 1:
        best_sim = -1.0
        bi_, bj_ = 0, 1
        for ci in range(len(clusters)):
            for cj in range(ci + 1, len(clusters)):
                # クラスタ間の平均類似度
                sims = []
                for k1 in clusters[ci]:
                    for k2 in clusters[cj]:
                        v1 = stream_vectors[k1]
                        v2 = stream_vectors[k2]
                        sims.append(cosine_sim(v1, v2))
                avg_sim = sum(sims) / len(sims) if sims else 0.0
                if avg_sim > best_sim:
                    best_sim = avg_sim
                    bi_, bj_ = ci, cj
        if best_sim < merge_threshold:
            break
        clusters[bi_] = clusters[bi_] + clusters[bj_]
        del clusters[bj_]

    # 各クラスタをTaskNodeに変換
    results: list[TaskNode] = []

    for cluster in clusters:
        # ブランチ名を収集
        branches: list[str] = []
        for proj, branch in cluster:
            if branch and branch != "__no_branch__":
                branches.append(branch)
            elif branch == "__no_branch__":
                branches.append(f"{proj}/main")

        # ブロック集合を統合
        block_set: set[int] = set()
        for skey in cluster:
            block_set |= stream_block_indices.get(skey, set())

        # Active時間を計算
        active = sum(
            (blocks[bi].end - blocks[bi].start).total_seconds() / 60
            for bi in block_set
            if bi < len(blocks)
        )

        # キーワード統合
        all_kws: Counter = Counter()
        for skey in cluster:
            for w in stream_kws.get(skey, []):
                all_kws[w] += 1
        # コンテキストキーワードを除外して上位を選定
        task_kws = [
            w for w, _ in all_kws.most_common(5)
            if w not in context_kws
        ]
        if not task_kws:
            task_kws = [w for w, _ in all_kws.most_common(3)]

        # ピーク時間
        peak_times: list[str] = []
        if block_set:
            sorted_blocks = sorted(block_set)
            # 最もActive時間の長いブロック上位3つのピーク
            block_durations = [
                (bi, (blocks[bi].end - blocks[bi].start).total_seconds())
                for bi in sorted_blocks if bi < len(blocks)
            ]
            block_durations.sort(key=lambda x: -x[1])
            for bi, _ in block_durations[:5]:
                mid = blocks[bi].start + (blocks[bi].end - blocks[bi].start) / 2
                peak_times.append(mid.strftime("%m/%d %H:%M"))

        results.append(
            TaskNode(
                keywords=task_kws,
                active_minutes=active,
                sigma_minutes=0.0,
                peak_times=peak_times,
                block_set=block_set,
                branches=sorted(set(branches)),
            )
        )

    results.sort(key=lambda t: -t.active_minutes)
    return results, context_kws
