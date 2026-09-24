import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.pilot.kld_eval import _eval_windows


def test_kld_uses_locked_regions_instead_of_prefix(tmp_path):
    region = tmp_path / "eval-regions.json"
    region.write_text(json.dumps({"regions": [
        {"name": "r0", "start_token": 40, "n_windows": 2},
    ]}))
    args = SimpleNamespace(
        regions=str(region), region_seq=10, ctx=10, chunks=2,
        allow_training_data=False, corpus="unused")
    ids = np.arange(100)
    windows, split = _eval_windows(ids, args, tmp_path / "student.pt")
    assert windows.tolist() == [list(range(40, 50)), list(range(50, 60))]
    assert split["split"] == "locked_eval_regions"
    assert split["ranges"][0]["start"] == 40


def test_kld_requires_explicit_contamination_opt_in(tmp_path):
    args = SimpleNamespace(
        regions=None, region_seq=10, ctx=10, chunks=1,
        allow_training_data=False, corpus="unused")
    with pytest.raises(SystemExit, match="allow-training-data"):
        _eval_windows(np.arange(20), args, tmp_path / "student.pt")
