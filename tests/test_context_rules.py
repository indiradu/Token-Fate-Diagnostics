from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from regret_remasking.context_rules import DIST_BINS, ContextRule, block_features, simple_payload
from regret_remasking.paper_decode import PAPER_ARMS, WINDOW_ARMS, PaperDecodeConfig, _validate
from test_frontier_arms import peaked
from test_paper_token_kl_commitment import _decode

MASK = 0


def _frames():
    # block 1 = positions 1-4, block 2 = positions 5-8; position 4 is weak, position 5 is very confident but in the next block
    return [{1: peaked(1, 0.99), 2: peaked(2, 0.9), 3: peaked(3, 0.85), 4: peaked(4, 0.3), 5: peaked(1, 0.999),
             6: peaked(2, 0.6), 7: peaked(3, 0.6), 8: peaked(4, 0.6)}]


def _writes(result):
    out: dict[int, list[int]] = {}
    for event in result.events:
        if event["type"] == "token_commit":
            out.setdefault(event["global_step"], []).append(event["position"])
    return out


def test_registration_and_validation() -> None:
    assert "X-block" in PAPER_ARMS and "X-block" not in WINDOW_ARMS
    with pytest.raises(ValueError):
        _validate(PaperDecodeConfig(arm="X-block"))                                   # needs a rule
    with pytest.raises(ValueError):
        _validate(PaperDecodeConfig(arm="C-block", block_rule=ContextRule("conf", 0.9)))  # only X-block takes a rule
    _validate(PaperDecodeConfig(arm="X-block", block_rule=ContextRule("conf", 0.9)))
    with pytest.raises(ValueError):
        ContextRule("conf", 1.5)
    with pytest.raises(ValueError):
        ContextRule("gbm", 0.5, features=["margin", "nonsense"])


@pytest.mark.parametrize("c", [0.95, 0.9, 0.8, 0.5])
def test_conf_rule_reproduces_c_block(c: float) -> None:
    base = dict(gen_length=8, block_length=4, mask_id=MASK)
    ref, _ = _decode(_frames(), PaperDecodeConfig(arm="C-block", threshold=c, **base))
    mine, _ = _decode(_frames(), PaperDecodeConfig(arm="X-block", block_rule=ContextRule("conf", c), **base))
    assert _writes(ref) == _writes(mine)
    assert ref.nfe == mine.nfe and list(ref.token_ids) == list(mine.token_ids)
    assert ref.fallback_count == mine.fallback_count and ref.adaptive_accept_count == mine.adaptive_accept_count


def test_rule_writes_only_inside_the_current_block_and_falls_back_to_best_score() -> None:
    rule = ContextRule("conf", 0.999)   # nothing clears it except position 5 (next block): never written early
    result, _ = _decode(_frames(), PaperDecodeConfig(arm="X-block", gen_length=8, block_length=4, block_rule=rule, mask_id=MASK))
    w = _writes(result)
    assert w[0] == [1] and w[1] == [2] and w[2] == [3] and w[3] == [4]   # one best-scoring blank per call
    assert min(p for s, v in w.items() if s >= 4 for p in v) >= 5
    assert result.final_mask_count == 0


def test_simple_rule_uses_the_distance_bin_table() -> None:
    # distance-1 blanks: score = confidence (identity table); far blanks (>8): score = confidence - 0.2
    ident = (np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    lower = (np.array([0.0, 1.0]), np.array([0.0, 0.8]))
    payload = {k: (lower if k == len(DIST_BINS) - 1 else ident) for k in range(len(DIST_BINS))}
    rule = ContextRule("simple", 0.7, payload=payload)
    feats = {"confidence": np.array([0.75, 0.75, 0.9]), "dist_written": np.array([1.0, 20.0, 20.0])}
    s = rule.scores(feats)
    assert s[0] == pytest.approx(0.75) and s[1] == pytest.approx(0.6) and s[2] == pytest.approx(0.72)
    assert (s >= rule.lam).tolist() == [True, False, True]


def test_simple_payload_matches_isotonic_predict() -> None:
    from sklearn.isotonic import IsotonicRegression
    rng = np.random.default_rng(0)
    x = rng.random(500)
    y = (rng.random(500) < x).astype(float)
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(x, y)
    xs, ys = simple_payload({0: iso})[0]
    q = rng.random(50)
    assert np.allclose(np.interp(q, xs, ys), iso.predict(q))


def test_gbm_rule_scores_a_named_feature_list_and_roundtrips(tmp_path) -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier
    rng = np.random.default_rng(1)
    X = rng.random((400, 2)).astype(np.float32)
    y = (X[:, 0] > 0.5).astype(int)
    model = HistGradientBoostingClassifier(max_iter=20, random_state=0).fit(X, y)
    rule = ContextRule("gbm", 0.5, features=["confidence", "margin"], payload=model, name="t")
    feats = {"confidence": np.array([0.9, 0.1], dtype=np.float32), "margin": np.array([0.3, 0.3], dtype=np.float32)}
    s = rule.scores(feats)
    assert s[0] > 0.5 > s[1]
    rule.save(tmp_path / "r.pkl")
    back = ContextRule.load(tmp_path / "r.pkl")
    assert np.allclose(back.scores(feats), s)


def test_block_features_match_the_collector_definitions() -> None:
    # sequence: 2 prompt tokens, answer of 8 slots; block_length 4, block 1 = answer slots 0-3 (absolute 2-5). Written: slot 1 and slot 6.
    prompt, gen = 2, 8
    x = torch.full((1, prompt + gen), MASK)
    x[0, :prompt] = torch.tensor([7, 8])
    x[0, prompt + 1] = 9
    x[0, prompt + 6] = 9
    eligible = torch.tensor([2, 4, 5])                 # absolute positions of the blanks left in block 1 (slot 1 already written)
    vocab = 6
    logp = torch.log(torch.tensor([peaked(1, 0.8), peaked(2, 0.5), peaked(3, 0.4)]))
    conf = torch.tensor([0.8, 0.5, 0.4])
    f = block_features(x, prompt, eligible, conf, logp, MASK, 4, gen)
    assert f["confidence"].tolist() == pytest.approx([0.8, 0.5, 0.4])
    # distance to the nearest written answer slot (slots 1 and 6): blank slot 0 -> 1, slot 2 -> 1, slot 3 -> 2
    assert f["dist_written"].tolist() == [1.0, 1.0, 2.0]
    # one slot written in the block so far: 4 - 3 blanks = 1 written; share 1/4
    assert f["pos_in_block"].tolist() == [0.25, 0.25, 0.25]
    p = torch.tensor(peaked(1, 0.8))
    assert f["entropy"][0] == pytest.approx(float(-(p * p.log()).sum()), rel=1e-5)
    assert f["margin"][0] == pytest.approx(0.8 - 0.04, rel=1e-5)
    # local mask ratio, window +-4, over the whole 10-token row: blank at absolute position 2 sees positions 0..6 (7 positions, of which 3 are written/prompt: 0,1 prompt... )
    row = (x[0] == MASK).float()
    expected = [float(row[max(0, a - 4):min(10, a + 5)].mean()) for a in (2, 4, 5)]
    assert f["local_mask_ratio"].tolist() == pytest.approx(expected)


def test_pos_in_block_is_clamped_to_one_step_at_the_first_call_of_a_block() -> None:
    prompt, gen = 1, 4
    x = torch.full((1, prompt + gen), MASK)
    x[0, 0] = 7
    eligible = torch.arange(1, 5)                      # all four blanks of the block still masked -> nothing written in the block
    logp = torch.log(torch.tensor([peaked(1, 0.5)] * 4))
    f = block_features(x, prompt, eligible, torch.full((4,), 0.5), logp, MASK, 4, gen)
    assert f["pos_in_block"].tolist() == [0.25] * 4 and f["dist_written"].tolist() == [4.0] * 4
