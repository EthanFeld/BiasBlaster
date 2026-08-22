from biasblaster.qkrylov_basis import (
    KrylovTimeCandidateAssessment,
    select_tfim_time_step,
)


def _assessment(time_step, rank, margin, residual=0.0, cost=100):
    return KrylovTimeCandidateAssessment(
        time_step=float(time_step),
        retained_rank=int(rank),
        dimension=2,
        robust_margin=float(margin),
        minimum_overlap_eigenvalue=0.1,
        maximum_overlap_eigenvalue=1.9,
        predicted_condition_number=19.0,
        conditioning_floor=0.0,
        first_order_residual_rmse=float(residual),
        weighted_two_qubit_executions=int(cost),
    )


def test_selector_uses_smallest_fully_resolvable_spacing(monkeypatch):
    candidates = {
        0.2: _assessment(0.2, 1, 0.5),
        0.4: _assessment(0.4, 2, 0.01),
        0.6: _assessment(0.6, 2, 0.20),
    }

    def fake_assess(value, **kwargs):
        return candidates[float(value)], f"plan-{value}"

    monkeypatch.setattr(
        "biasblaster.qkrylov_basis.assess_tfim_time_step", fake_assess
    )
    selection = select_tfim_time_step([0.2, 0.4, 0.6])
    assert selection.selected_time_step == 0.4
    assert selection.selected_plan == "plan-0.4"


def test_selector_fallback_does_not_use_finite_channel_residual(monkeypatch):
    # Both candidates fail full-rank resolution and are otherwise equivalent.
    # The earlier candidate has a deliberately much worse offline residual; the
    # residual is a regression diagnostic and must not influence selection.
    candidates = {
        0.2: _assessment(0.2, 1, 0.05, residual=100.0, cost=100),
        0.4: _assessment(0.4, 1, 0.05, residual=0.0, cost=100),
    }

    def fake_assess(value, **kwargs):
        return candidates[float(value)], f"plan-{value}"

    monkeypatch.setattr(
        "biasblaster.qkrylov_basis.assess_tfim_time_step", fake_assess
    )
    selection = select_tfim_time_step([0.2, 0.4])
    assert selection.selected_time_step == 0.2
