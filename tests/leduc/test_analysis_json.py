"""The visualizer's data contract: what `leduc-analysis --json` promises."""

import json

from drawmaha_solver.leduc.analysis import LP_VALUE_P0, run, to_json, write_json
from drawmaha_solver.leduc.game import ACTION_SYMBOL, all_infosets


def test_export_covers_every_infoset_with_a_distribution(tmp_path):
    trajectory = run(5, checkpoints=2)
    out = to_json(trajectory)

    assert set(out["strategy"]) == {str(s) for s in all_infosets()}
    assert len(out["strategy"]) == 288

    legal = {str(s): {ACTION_SYMBOL[a] for a in s.legal_actions()} for s in all_infosets()}
    for key, row in out["strategy"].items():
        assert set(row) == legal[key]
        assert abs(sum(row.values()) - 1.0) < 1e-9

    assert out["iterations"] == 5
    assert out["gameValueExact"] == LP_VALUE_P0
    assert out["exploitabilityAverage"] > 0
    assert out["exploitabilityCurrent"] > 0

    path = tmp_path / "solve.json"
    write_json(trajectory, path)
    assert json.loads(path.read_text()) == out
