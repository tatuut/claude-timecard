"""build_tfidf, _tokenize, extract_keywords, stopwords/synonyms管理."""

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

from ..config import C_GREEN, C_RESET

# --- Stopwords ---
_STOPWORDS_FILE = Path(__file__).resolve().parent.parent.parent / "stopwords.json"


def _load_stopwords() -> set[str]:
    """stopwords.json からストップワードを読み込む."""
    if not _STOPWORDS_FILE.exists():
        return set()
    with open(_STOPWORDS_FILE, encoding="utf-8") as f:
        data = json.load(f)
    words: set[str] = set()
    for key, vals in data.items():
        if key.startswith("_"):
            continue
        words.update(vals)
    return words


def _add_stopwords(new_words: list[str], category: str = "ユーザー追加") -> None:
    """stopwords.json にストップワードを追加."""
    if _STOPWORDS_FILE.exists():
        with open(_STOPWORDS_FILE, encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {"_comment": "カテゴリ別ストップワード"}

    existing = data.get(category, [])
    merged = sorted(set(existing) | set(new_words))
    data[category] = merged

    with open(_STOPWORDS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  {C_GREEN}stopwords.json に {len(new_words)} 語追加 → {category}{C_RESET}")


_GRAMMAR_STOPS = _load_stopwords()

# --- Synonyms ---
# 同義語は2箇所から読む:
# 1. リポジトリ同梱のデフォルト（synonyms.json — 汎用的なもののみ）
# 2. ユーザーローカル（~/.config/claude-timecard/synonyms.json — 個人/業務固有）
_SYNONYMS_DEFAULT = Path(__file__).resolve().parent.parent.parent / "synonyms.json"
_SYNONYMS_USER = Path.home() / ".config" / "claude-timecard" / "synonyms.json"


def _load_synonyms() -> dict[str, str]:
    """同義語辞書を読み込む（デフォルト + ユーザーローカルをマージ）."""
    mapping: dict[str, str] = {}
    for path in [_SYNONYMS_DEFAULT, _SYNONYMS_USER]:
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for canonical, aliases in data.items():
            if canonical.startswith("_"):
                continue
            for alias in aliases:
                mapping[alias] = canonical
    return mapping


def import_sudachi_synonyms(path: str, min_group_size: int = 2) -> int:
    """Sudachi同義語辞書CSVをインポートしてユーザーローカルに保存.

    Args:
        path: synonyms.txt のパス（またはURL）
        min_group_size: 最小グループサイズ（1語のグループは無視）

    Returns:
        インポートされたグループ数
    """
    import csv
    from io import StringIO

    # ファイル読み込み（URLまたはローカルパス）
    if path.startswith("http://") or path.startswith("https://"):
        import urllib.request
        resp = urllib.request.urlopen(path, timeout=30)
        text = resp.read().decode("utf-8")
    else:
        with open(path, encoding="utf-8") as f:
            text = f.read()

    # グループごとにパース
    groups: dict[str, list[tuple[str, int]]] = {}  # group_id -> [(word, word_type)]
    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(",")
        if len(parts) < 9:
            continue
        group_id = parts[0]
        word_type = int(parts[4]) if parts[4] else 0  # 0=代表語
        word = parts[8]
        if not word:
            continue
        if group_id not in groups:
            groups[group_id] = []
        groups[group_id].append((word, word_type))

    # ユーザーローカル辞書にマージ
    _SYNONYMS_USER.parent.mkdir(parents=True, exist_ok=True)
    if _SYNONYMS_USER.exists():
        with open(_SYNONYMS_USER, encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}

    imported = 0
    for group_id, words in groups.items():
        if len(words) < min_group_size:
            continue
        # 代表語（word_type=0）を正規形とする
        canonical_candidates = [w for w, t in words if t == 0]
        canonical = canonical_candidates[0] if canonical_candidates else words[0][0]
        aliases = [w for w, _ in words if w != canonical]
        if not aliases:
            continue
        existing = set(data.get(canonical, []))
        existing.update(aliases)
        data[canonical] = sorted(existing)
        imported += 1

    with open(_SYNONYMS_USER, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return imported


def add_synonym(canonical: str, aliases: list[str]) -> None:
    """ユーザーローカルの同義語辞書に追加."""
    _SYNONYMS_USER.parent.mkdir(parents=True, exist_ok=True)
    if _SYNONYMS_USER.exists():
        with open(_SYNONYMS_USER, encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}
    existing = data.get(canonical, [])
    data[canonical] = sorted(set(existing) | set(aliases))
    with open(_SYNONYMS_USER, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    # グローバルマップも更新
    for alias in aliases:
        _SYNONYM_MAP[alias] = canonical


_SYNONYM_MAP = _load_synonyms()

# --- ユーザー名の動的除外 ---
# OSのユーザー名をストップワードに自動追加（トークンに出ないように）
import os as _os
_USERNAME = _os.getenv("USERNAME") or _os.getenv("USER") or ""
if _USERNAME and len(_USERNAME) >= 3:
    _GRAMMAR_STOPS.add(_USERNAME.lower())


def tokenize_with_extractors(
    text: str,
    extractors: list[str] | None = None,
    **context,
) -> list[str]:
    """Registry登録済みの抽出器でトークン化.

    Args:
        text: 対象テキスト
        extractors: 使う抽出器名のリスト。Noneなら ["unigram"] のみ。
        context: 追加コンテキスト（branch名等）
    """
    from .extractors import get_all_extractors

    if extractors is None:
        extractors = ["unigram"]

    all_extractors = get_all_extractors()
    tokens: list[str] = []
    for name in extractors:
        if name in all_extractors:
            ext = all_extractors[name]()
            tokens.extend(ext.extract(text, **context))
    return tokens


def _tokenize(text: str, **context) -> list[str]:
    """テキストからトークンを抽出。Registry経由で全登録済み抽出器を使用。

    後方互換: 既存の全呼び出し元はこの関数を使い続ける。
    内部でtokenize_with_extractorsに委譲（unigram + bigram）。
    """
    return tokenize_with_extractors(text, extractors=["unigram", "bigram"], **context)


def build_tfidf(blocks: list) -> dict[str, float]:
    """全ブロックからTF-IDFスコアを算出."""
    n_blocks = len(blocks)
    if n_blocks == 0:
        return {}

    # 各ブロックの語彙を計算
    block_vocabs: list[Counter] = []
    for b in blocks:
        vocab: Counter = Counter()
        for msg in b.messages:
            for token in _tokenize(msg):
                vocab[token] += 1
        block_vocabs.append(vocab)

    # TF（全体合算）
    tf: Counter = Counter()
    for v in block_vocabs:
        tf.update(v)

    # DF（何ブロックに出現したか）
    df: Counter = Counter()
    for v in block_vocabs:
        for word in v:
            df[word] += 1

    # TF-IDF
    tfidf: dict[str, float] = {}
    for word in tf:
        idf = math.log(n_blocks / df[word]) if df[word] < n_blocks else 0.0
        tfidf[word] = tf[word] * idf

    return tfidf


def extract_keywords(
    messages: list[str],
    top_n: int = 5,
    tfidf_scores: dict[str, float] | None = None,
) -> list[tuple[str, int]]:
    """メッセージ群からキーワードを抽出."""
    word_counts: Counter = Counter()
    for msg in messages:
        for token in _tokenize(msg):
            word_counts[token] += 1

    if tfidf_scores:
        scored = []
        for word, count in word_counts.items():
            score = tfidf_scores.get(word, 0.0)
            if score > 0:
                scored.append((word, count, score))
        scored.sort(key=lambda x: (-x[2], -x[1]))
        return [(w, c) for w, c, _ in scored[:top_n]]
    else:
        return word_counts.most_common(top_n)
