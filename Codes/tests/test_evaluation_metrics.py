import numpy as np
import pytest

from evaluation.metrics import evaluate_predictions, plot_confusion_matrix


def test_perfect_predictions_score_1():
    labels = ["corn", "soybean", "wheat"]
    y_true = ["corn", "soybean", "wheat", "corn", "wheat"]
    y_pred = ["corn", "soybean", "wheat", "corn", "wheat"]

    result = evaluate_predictions(y_true, y_pred, labels)
    assert result["accuracy"] == pytest.approx(1.0)
    assert result["macro_f1"] == pytest.approx(1.0)
    assert result["weighted_f1"] == pytest.approx(1.0)
    assert result["confusion_matrix"].sum() == len(y_true)
    assert np.trace(result["confusion_matrix"]) == len(y_true)


def test_all_wrong_predictions_score_0_accuracy():
    labels = ["corn", "soybean"]
    y_true = ["corn", "corn", "soybean", "soybean"]
    y_pred = ["soybean", "soybean", "corn", "corn"]

    result = evaluate_predictions(y_true, y_pred, labels)
    assert result["accuracy"] == pytest.approx(0.0)
    assert result["macro_f1"] == pytest.approx(0.0)


def test_confusion_matrix_shape_matches_label_count():
    labels = ["corn", "soybean", "wheat"]
    y_true = ["corn", "soybean", "wheat"]
    y_pred = ["corn", "wheat", "wheat"]

    result = evaluate_predictions(y_true, y_pred, labels)
    assert result["confusion_matrix"].shape == (3, 3)


def test_plot_confusion_matrix_writes_a_file(tmp_path):
    labels = ["corn", "wheat"]
    cm = np.array([[5, 1], [2, 4]])
    out_path = plot_confusion_matrix(cm, labels, "Test", tmp_path / "cm.png")
    assert out_path.exists()
    assert out_path.stat().st_size > 0
