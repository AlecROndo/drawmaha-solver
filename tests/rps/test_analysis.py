import numpy as np

from drawmaha_solver.rps.analysis import run_self_play, run_vs_fixed

def test_run_self_play_shapes_and_convergence():
    traj = run_self_play(5_000, seed=0)
    assert traj.current0.shape == (5_000, 3)
    assert traj.average0.shape == (5_000, 3)
    assert traj.exploitability0.shape == (5_000,)
    # every recorded average is a distribution
    assert np.allclose(traj.average0.sum(axis=1), 1.0)
    # the late average is closer to Nash than the early average
    assert traj.exploitability0[-1] < traj.exploitability0[100]

def test_run_vs_fixed_learns_the_counter():
    traj = run_vs_fixed(5_000, opponent=np.array([0.5, 0.25, 0.25]), seed=0)
    assert traj.average.shape == (5_000, 3)
    assert traj.payoffs.shape == (5_000,)
    assert traj.average[-1, 1] > 0.8  # paper

def test_main_writes_four_figures(tmp_path, monkeypatch):
    import sys

    from drawmaha_solver.rps import analysis

    monkeypatch.setattr(
        sys, "argv", ["rps-analysis", "--iters", "2000", "--out", str(tmp_path)]
    )
    analysis.main()
    written = {p.name for p in tmp_path.glob("*.png")}
    assert written == {
        "self_play_average_strategy.png",
        "self_play_current_vs_average.png",
        "self_play_exploitability.png",
        "vs_biased_average_strategy.png",
    }
