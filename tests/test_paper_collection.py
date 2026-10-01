from pathlib import Path
from types import SimpleNamespace
import json
import random

import numpy as np
import pytest
import torch

from regret_remasking import paper_collection as collection


def test_retained_steps_are_predeclared_and_include_endpoints():
    a = collection.retained_steps(128, .3, random.Random(23))
    assert len(a) == 39 and a[0] == 0 and a[-1] == 127
    assert a == collection.retained_steps(128, .3, random.Random(23))


def test_quality_filter_checks_generated_code_without_running_it():
    entry = {"family": "code", "question": 'def f(x):\n    """double x"""\n', "metadata": {"entry_point": "f", "test": "def check(candidate):\n    assert candidate(2)==4\n"}}
    assert collection.trace_quality_filter("def f(x):\n    return x * 2", entry) == (True, [])
    assert "invalid_code_syntax" in collection.trace_quality_filter("def f(x):\n    if", entry)[1]
    assert collection.trace_quality_filter("a b c d " * 30, {"family": "qa"})[0] is False


def test_shared_storage_cap_is_atomic_and_idempotent(tmp_path):
    path = tmp_path / "ledger.json"
    collection.reserve_storage(path, "a", 100, 200 / 1024**3)
    collection.reserve_storage(path, "a", 100, 200 / 1024**3)
    collection.reserve_storage(path, "b", 100, 200 / 1024**3)
    assert json.loads(path.read_text())["bytes"] == 200
    with pytest.raises(RuntimeError, match="cap"):
        collection.reserve_storage(path, "c", 1, 200 / 1024**3)
    with pytest.raises(ValueError, match="changed"):
        collection.reserve_storage(path, "a", 99, 200 / 1024**3)


class Projection:
    def encode_hidden(self, hidden):
        return torch.ones((*hidden[0].shape[:2], 832))


def test_hidden_labels_are_backfilled_and_full_context_is_preserved(tmp_path, monkeypatch):
    entry = {"record_id": "fit:math:e", "example_id": "e", "source_id": "source:e", "family": "math", "dataset": "gsm8k", "split": "fit", "prompt_text": "p", "question": "q", "gold_answer": "4", "prompt_hash": "ph", "metadata": {}, "teacher_config": {"steps": 4, "gen_length": 4, "block_length": 2, "retained_step_fraction": .5}}
    manifest = tmp_path / "fit.jsonl";manifest.write_text(json.dumps(entry)+"\n")
    checkpoint = tmp_path / "projection.pt";checkpoint.write_bytes(b"fake fixture")
    monkeypatch.setattr(collection.HiddenProjector, "load", lambda *a, **k: Projection())
    monkeypatch.setattr(collection, "score_paper_generation", lambda *a: (True, "passed"))
    calls = []
    def decode(*args, step_observer=None, **kwargs):
        calls.append(1)
        for step in range(4):
            ids = torch.tensor([[1, 2, 31, 31, 31, 31]])
            states = torch.tensor([[0, 0, 2, 2, 2, 2]], dtype=torch.uint8)
            proposals = torch.tensor([[1, 2, 9, 6, 9, 8]])
            if step == 3:
                ids[0, 2:5] = torch.tensor([5, 6, 7]);states[0, 2:5] = 1
            step_observer({"step": step, "input_ids": ids, "states": states, "proposal_tokens": proposals,
                           "proposal_positions": torch.arange(6), "prompt_length": 2,
                           "last_hidden": (torch.zeros(1,6,4),)*3})
        return SimpleNamespace(nfe=4, final_mask_count=0, immutable_token_violations=0, text="reasoning #### 4", token_ids=[5,6,7,8], prompt_length=2)
    monkeypatch.setattr(collection, "decode_paper_policy", decode)
    out = collection.collect_hidden_manifest(manifest,tmp_path/'out',tmp_path/'model',checkpoint,device='cpu',model=object(),tokenizer=object(),storage_ledger=tmp_path/'ledger.json')
    assert out["complete"] and out["accepted_count"] == 1
    path = Path(out["examples"][0]["example_dir"])
    features = np.load(path/'features.npy');labels = np.load(path/'labels.npy');states=np.load(path/'states.npy')
    assert features.shape == (2,6,832)
    assert labels[0].tolist() == [1,1,0,1,0,1]
    assert states[0,4] == 2  # includes a future-block target, not just teacher currentblock
    again = collection.collect_hidden_manifest(manifest,tmp_path/'out',tmp_path/'model',checkpoint,device='cpu',model=object(),tokenizer=object(),storage_ledger=tmp_path/'ledger.json')
    assert again == out and len(calls) == 1
    with (path/'features.npy').open('ab') as f:f.write(b'corruption')
    with pytest.raises(ValueError,match="hash mismatch"):
        collection.verify_trace_files(path,collection.file_hash(checkpoint))
