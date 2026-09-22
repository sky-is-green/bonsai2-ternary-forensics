"""Decision-margin probe metrics (pure numpy)."""

from __future__ import annotations

import numpy as np
import pytest

from bonsai_forensics import margin


def test_identical_distributions_show_no_drift():
    logits = np.array([[1.0, 0.0, -1.0], [0.5, 0.4, 0.1]])
    s = margin.summarize(margin.compare(logits, logits))
    assert s["kl"]["mean"] == pytest.approx(0.0, abs=1e-9)
    assert s["margin_delta"]["mean"] == pytest.approx(0.0, abs=1e-9)
    assert s["top1_agreement"] == 1.0
    assert s["overall"]["margin_retention"] == pytest.approx(1.0)


def test_sharper_student_gains_margin():
    teacher = np.array([[1.0, 0.0, -1.0]])
    sharper = teacher * 2.0
    s = margin.summarize(margin.compare(teacher, sharper))
    assert s["overall"]["margin_retention"] > 1.0
    assert s["entropy_student"]["mean"] < s["entropy_teacher"]["mean"]


def test_flatter_student_loses_margin():
    teacher = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0]])
    flatter = teacher * 0.5
    s = margin.summarize(margin.compare(teacher, flatter))
    assert s["overall"]["margin_retention"] < 1.0
    assert s["entropy_student"]["mean"] > s["entropy_teacher"]["mean"]
    assert s["frac_positions_less_decisive"] > 0.5


def test_top1_disagreement_is_detected():
    teacher = np.array([[0.30, 0.29, 0.0]])
    student = np.array([[0.29, 0.30, 0.0]])  # flips the top choice
    s = margin.summarize(margin.compare(teacher, student))
    assert s["top1_agreement"] == 0.0


def test_decisive_positions_can_drift_more_than_average():
    # Nine marginal positions are preserved exactly; one confident position
    # collapses. The decisive stratum must show the damage the mean hides.
    low = np.array([1.0, 0.0, 0.0])
    teacher = np.tile(low, (10, 1))
    teacher[9] = np.array([5.0, 0.0, 0.0])
    student = np.tile(low, (10, 1))
    student[9] = low  # the confident decision loses its margin entirely

    s = margin.summarize(margin.compare(teacher, student), decisive_quantile=0.9)
    assert s["decisive"]["n"] < s["overall"]["n"]
    assert s["decisive"]["margin_retention"] < s["overall"]["margin_retention"]
    assert s["decisive"]["margin_retention"] == pytest.approx(0.364 / 0.990, abs=0.01)


def test_shape_mismatch_and_tiny_vocab_are_errors():
    with pytest.raises(ValueError, match="shape mismatch"):
        margin.compare(np.zeros((2, 3)), np.zeros((3, 3)))
    with pytest.raises(ValueError, match="two vocabulary"):
        margin.compare(np.zeros((2, 1)), np.zeros((2, 1)))


def test_summarize_rejects_empty():
    empty = {"margin_teacher": np.array([]), "margin_student": np.array([]),
             "kl": np.array([]), "entropy_teacher": np.array([]),
             "entropy_student": np.array([]), "agree": np.array([])}
    with pytest.raises(ValueError, match="no positions"):
        margin.summarize(empty)
