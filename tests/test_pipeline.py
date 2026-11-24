import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from predict import recommend, validate_inputs

RICE = dict(N=90, P=42, K=43, temperature=20.9, humidity=82.0, ph=6.5, rainfall=203.0)


def test_valid_input_has_no_errors():
    assert validate_inputs(RICE) == []


def test_out_of_range_rejected():
    assert validate_inputs({**RICE, "ph": 15})


def test_prediction_returns_ranked_probabilities():
    ranked = recommend(RICE, top_k=3)
    assert len(ranked) == 3
    assert ranked[0][1] >= ranked[1][1] >= ranked[2][1]
    assert ranked[0][0] == "rice"


def test_invalid_raises():
    with pytest.raises(ValueError):
        recommend({**RICE, "N": -5})
