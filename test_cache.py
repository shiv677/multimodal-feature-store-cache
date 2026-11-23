import pytest
from pathlib import Path
import json

try:
    from .cache import SemanticVectorCache, cosine_similarity
    from .models import EvictionPolicy, VectorRecord
    from .main import run_benchmark
except ImportError:
    from cache import SemanticVectorCache, cosine_similarity
    from models import EvictionPolicy, VectorRecord
    from main import run_benchmark


def test_cosine_similarity():
    v1 = [1.0, 0.0, 0.0]
    v2 = [1.0, 0.0, 0.0]
    v3 = [0.0, 1.0, 0.0]
    v4 = [0.7071, 0.7071, 0.0]

    assert cosine_similarity(v1, v2) == pytest.approx(1.0, abs=1e-4)
    assert cosine_similarity(v1, v3) == pytest.approx(0.0, abs=1e-4)
    assert cosine_similarity(v1, v4) == pytest.approx(0.7071, abs=1e-3)
    assert cosine_similarity([0.0, 0.0], [0.0, 0.0]) == 0.0


def test_exact_match_hit():
    cache = SemanticVectorCache(dim=4, num_tables=2, num_bits=4)
    rec = VectorRecord(
        id="rec_1",
        prompt="Describe cat on windowsill",
        vector=[0.5, 0.5, 0.5, 0.5],
        response="A fluffy orange cat resting on a sunny windowsill.",
        token_count=120,
        cost=0.0024,
    )
    cache.put(rec)

    match = cache.query(vector=[0.5, 0.5, 0.5, 0.5], prompt="Describe cat on windowsill")
    assert match.cache_hit is True
    assert match.record is not None
    assert match.record.id == "rec_1"
    assert match.similarity == pytest.approx(1.0, abs=1e-4)
    assert match.match_type == "EXACT"

    metrics = cache.get_metrics()
    assert metrics.cache_hits == 1
    assert metrics.saved_tokens == 120
    assert metrics.estimated_cost_saved == pytest.approx(0.0024, abs=1e-5)


def test_semantic_near_duplicate_hit():
    cache = SemanticVectorCache(dim=4, similarity_threshold=0.85)
    rec = VectorRecord(
        id="rec_med",
        prompt="Pneumonia X-ray finding",
        vector=[0.6, 0.8, 0.0, 0.0],
        response="Consolidation in left lobe.",
        token_count=200,
    )
    cache.put(rec)

    # Slightly perturbed vector: cosine sim should be around ~0.99
    perturbed_query = [0.61, 0.79, 0.05, 0.0]
    match = cache.query(vector=perturbed_query)

    assert match.cache_hit is True
    assert match.record is not None
    assert match.record.id == "rec_med"
    assert match.similarity >= 0.85
    assert match.match_type in ("SEMANTIC", "EXACT")


def test_cache_miss_dissimilar_vector():
    cache = SemanticVectorCache(dim=4, similarity_threshold=0.85)
    rec = VectorRecord(
        id="rec_audio",
        prompt="Speech transcription",
        vector=[1.0, 0.0, 0.0, 0.0],
        response="Hello world audio text.",
        token_count=50,
    )
    cache.put(rec)

    # Orthogonal / opposite vector
    query_vector = [0.0, 1.0, 0.0, 0.0]
    match = cache.query(vector=query_vector)

    assert match.cache_hit is False
    assert match.record is None
    assert match.similarity == pytest.approx(0.0, abs=1e-4)
    assert match.match_type == "MISS"

    metrics = cache.get_metrics()
    assert metrics.cache_misses == 1
    assert metrics.hit_rate == 0.0


def test_lru_memory_clamp_eviction():
    # Set a small memory clamp that only accommodates 2 records
    small_clamp = 1100
    cache = SemanticVectorCache(
        dim=4,
        max_memory_bytes=small_clamp,
        eviction_policy=EvictionPolicy.LRU,
    )

    rec1 = VectorRecord(id="r1", prompt="Prompt 1", vector=[0.1, 0.2, 0.3, 0.4], response="Response 1", token_count=10)
    rec2 = VectorRecord(id="r2", prompt="Prompt 2", vector=[0.2, 0.3, 0.4, 0.5], response="Response 2", token_count=20)
    rec3 = VectorRecord(id="r3", prompt="Prompt 3", vector=[0.3, 0.4, 0.5, 0.6], response="Response 3", token_count=30)

    cache.put(rec1)
    cache.put(rec2)
    assert cache.size() == 2

    # Query rec1 to make rec2 least recently used
    cache.query(vector=[0.1, 0.2, 0.3, 0.4], prompt="Prompt 1")

    # Ingesting rec3 should trigger eviction of rec2 (LRU)
    cache.put(rec3)

    assert cache.size() <= 2
    metrics = cache.get_metrics()
    assert metrics.eviction_count >= 1
    assert "r2" not in cache.records
    assert "r3" in cache.records


def test_fifo_and_lfu_eviction():
    # FIFO test
    fifo_cache = SemanticVectorCache(dim=2, max_memory_bytes=1000, eviction_policy="FIFO")
    rec1 = VectorRecord(id="f1", prompt="P1", vector=[1.0, 0.0], response="R1")
    rec2 = VectorRecord(id="f2", prompt="P2", vector=[0.0, 1.0], response="R2")
    rec3 = VectorRecord(id="f3", prompt="P3", vector=[0.5, 0.5], response="R3")

    fifo_cache.put(rec1)
    fifo_cache.put(rec2)
    fifo_cache.put(rec3)
    assert "f1" not in fifo_cache.records  # f1 arrived first

    # LFU test
    lfu_cache = SemanticVectorCache(dim=2, max_memory_bytes=1000, eviction_policy="LFU")
    lfu1 = VectorRecord(id="l1", prompt="P1", vector=[1.0, 0.0], response="R1")
    lfu2 = VectorRecord(id="l2", prompt="P2", vector=[0.0, 1.0], response="R2")
    lfu3 = VectorRecord(id="l3", prompt="P3", vector=[0.5, 0.5], response="R3")

    lfu_cache.put(lfu1)
    lfu_cache.put(lfu2)
    # Access lfu1 multiple times
    lfu_cache.query(vector=[1.0, 0.0], prompt="P1")
    lfu_cache.query(vector=[1.0, 0.0], prompt="P1")
    lfu_cache.put(lfu3)
    assert "l2" not in lfu_cache.records  # l2 had lowest access count


def test_dimension_validation():
    cache = SemanticVectorCache(dim=4)
    with pytest.raises(ValueError):
        cache.put(VectorRecord(id="bad_dim", prompt="", vector=[1.0, 2.0], response=""))

    with pytest.raises(ValueError):
        cache.query(vector=[1.0, 2.0, 3.0])


def test_same_prompt_orthogonal_vector_misses_and_oversized_record_rejected():
    cache = SemanticVectorCache(dim=4, similarity_threshold=0.85, max_memory_bytes=1200)
    rec = VectorRecord(
        id="rec_xray",
        prompt="Describe this X-ray",
        vector=[1.0, 0.0, 0.0, 0.0],
        response="Normal chest radiograph.",
    )
    cache.put(rec)

    # Same text prompt but orthogonal visual vector must not false-hit
    match = cache.query(vector=[0.0, 1.0, 0.0, 0.0], prompt="Describe this X-ray")
    assert match.cache_hit is False

    # Oversized record (> max_memory_bytes) must raise ValueError and not evict existing records
    oversized = VectorRecord(
        id="rec_huge",
        prompt="Huge",
        vector=[1.0, 0.0, 0.0, 0.0],
        response="X" * 2000,
    )
    with pytest.raises(ValueError):
        cache.put(oversized)
    assert "rec_xray" in cache.records
    assert cache.current_memory_bytes <= cache.max_memory_bytes


def test_lsh_1bit_neighbor_probe_above_200_records():
    cache = SemanticVectorCache(dim=4, num_tables=1, num_bits=4, similarity_threshold=0.85)
    # Populate > 200 records so linear scan fallback is disabled
    for i in range(210):
        cache.put(VectorRecord(id=f"fill_{i}", prompt=f"P{i}", vector=[1.0, float(i + 1), 0.0, 0.0], response="R"))

    plane0 = cache.hyperplanes[0][0]
    norm = sum(x * x for x in plane0) ** 0.5
    u = [x / norm for x in plane0]
    # Pick orthogonal unit vector w
    raw_w = [-u[1], u[0], 0.0, 0.0]
    w_norm = sum(x * x for x in raw_w) ** 0.5
    w = [x / w_norm for x in raw_w]

    # v_pos and v_neg slightly straddle plane0 (1-bit flip) while remaining >0.99 cosine similar
    v_pos = [w[j] + 0.02 * u[j] for j in range(4)]
    v_neg = [w[j] - 0.02 * u[j] for j in range(4)]

    cache.put(VectorRecord(id="target_neighbor", prompt="Target", vector=v_pos, response="Hit"))
    match = cache.query(vector=v_neg)
    assert match.cache_hit is True
    assert match.record is not None
    assert match.record.id == "target_neighbor"


def test_run_benchmark_end_to_end():
    sample_file = Path(__file__).parent / "sample_embeddings.json"
    metrics = run_benchmark(sample_file, similarity_threshold=0.85)

    assert metrics.total_queries > 0
    assert metrics.cache_hits >= 2
    assert metrics.cache_misses >= 1
    assert metrics.saved_tokens > 0
    assert metrics.hit_rate > 0.0
