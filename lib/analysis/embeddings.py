"""embedding取得, synonym merge."""

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from ..config import C_DIM, C_RESET

_EMBED_CACHE_DIR = Path.home() / ".cache" / "claude-timecard" / "embeddings"


def _vec_dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _vec_norm(a: list[float]) -> float:
    return math.sqrt(sum(x * x for x in a))


def cosine_sim(a: list[float], b: list[float]) -> float:
    """コサイン類似度."""
    na, nb = _vec_norm(a), _vec_norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return _vec_dot(a, b) / (na * nb)


def overlap_coeff(a: list[float], b: list[float]) -> float:
    """bのうちaの活動範囲と重なる割合 (0-1)."""
    total_b = sum(b)
    if total_b == 0:
        return 0.0
    return sum(min(x, y) for x, y in zip(a, b)) / total_b


def get_embeddings(words: list[str]) -> dict[str, list[float]] | None:
    """単語リストのembeddingを取得.

    LM Studio API (localhost) → fastembed → None のフォールバック。
    """
    # 1) LM Studio (OpenAI互換API)
    try:
        import urllib.request

        payload = json.dumps(
            {"model": "text-embedding-qwen3-embedding-8b", "input": words}
        ).encode()
        req = urllib.request.Request(
            "http://localhost:1236/v1/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
        result: dict[str, list[float]] = {}
        for item in data["data"]:
            result[words[item["index"]]] = item["embedding"]
        return result
    except Exception:
        pass

    # 2) fastembed (ONNX)
    try:
        from fastembed import TextEmbedding

        model = TextEmbedding(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        embs = list(model.embed(words))
        return dict(zip(words, [e.tolist() for e in embs]))
    except Exception:
        pass

    return None


def merge_synonyms_by_embedding(
    keywords: list[str],
    block_word_counts: list[Counter],
    threshold: float = 0.8,
) -> list[str]:
    """embeddingでキーワード間の同義語を検出しマージ."""
    if len(keywords) < 2:
        return keywords

    # embedding結果キャッシュ（ファイル）
    cache_key = hashlib.sha256("|".join(sorted(keywords)).encode()).hexdigest()[:16]
    cache_path = _EMBED_CACHE_DIR / f"syn_{cache_key}.json"

    merged_map: dict[str, str] | None = None
    if cache_path.exists():
        try:
            with open(cache_path, encoding="utf-8") as f:
                merged_map = json.load(f)
        except Exception:
            pass

    if merged_map is None:
        embs = get_embeddings(keywords)
        if embs is None:
            return keywords

        merged_map = {}
        used = set()
        for i, w1 in enumerate(keywords):
            if w1 in used:
                continue
            for w2 in keywords[i + 1 :]:
                if w2 in used:
                    continue
                sim = cosine_sim(embs[w1], embs[w2])
                if sim >= threshold:
                    tf1 = sum(wc.get(w1, 0) for wc in block_word_counts)
                    tf2 = sum(wc.get(w2, 0) for wc in block_word_counts)
                    if tf1 >= tf2:
                        merged_map[w2] = w1
                        used.add(w2)
                    else:
                        merged_map[w1] = w2
                        used.add(w1)
                        break

        _EMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(merged_map, f, ensure_ascii=False)

    if merged_map:
        merged_names = [f"{v}←{k}" for k, v in merged_map.items()]
        print(f"  {C_DIM}embedding同義語マージ: {', '.join(merged_names)}{C_RESET}")

    result = [w for w in keywords if w not in merged_map]
    return result
