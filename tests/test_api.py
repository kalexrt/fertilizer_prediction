from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
from fastapi.testclient import TestClient

from src.api import app


SAMPLE_750000 = {
    "temperature": 31,
    "humidity": 70,
    "moisture": 52,
    "soil_type": "Sandy",
    "crop_type": "Wheat",
    "nitrogen": 34,
    "potassium": 11,
    "phosphorus": 24,
}
RANKING_750000 = ["10-26-26", "DAP", "28-28"]


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def predict_ranking(client, sample):
    response = client.post("/predict", json=sample)
    assert response.status_code == 200, response.text
    return [prediction["fertilizer"] for prediction in response.json()]


def test_predict_matches_submission_for_row_750000(client):
    assert predict_ranking(client, SAMPLE_750000) == RANKING_750000


def assert_field_error(response, field, error_type):
    assert response.status_code == 422, response.text
    errors = response.json()["detail"]
    assert any(
        error["loc"] == ["body", field] and error["type"] == error_type
        for error in errors
    ), errors


@pytest.mark.parametrize("field", SAMPLE_750000)
def test_predict_rejects_missing_field(client, field):
    sample = SAMPLE_750000.copy()
    del sample[field]
    assert_field_error(client.post("/predict", json=sample), field, "missing")


@pytest.mark.parametrize(
    ("field", "value", "error_type"),
    [
        ("temperature", "hot", "float_parsing"),
        ("nitrogen", None, "float_type"),
        ("soil_type", 42, "string_type"),
        ("crop_type", "   ", "string_too_short"),
        ("moisture", "1e400", "finite_number"),
        ("temperature", "-1e400", "finite_number"),
        ("unknown_field", 0, "extra_forbidden"),
    ],
)
def test_predict_rejects_invalid_field(client, field, value, error_type):
    sample = {**SAMPLE_750000, field: value}
    assert_field_error(client.post("/predict", json=sample), field, error_type)


def test_parallel_predictions_match_each_samples_submission(client):
    sample_750001 = {
        "temperature": 27,
        "humidity": 62,
        "moisture": 45,
        "soil_type": "Red",
        "crop_type": "Sugarcane",
        "nitrogen": 30,
        "potassium": 14,
        "phosphorus": 15,
    }
    start = Barrier(2)

    def predict_together(sample):
        start.wait(timeout=10)
        return predict_ranking(client, sample)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(predict_together, [SAMPLE_750000, sample_750001]))

    assert results == [RANKING_750000, ["17-17-17", "20-20", "Urea"]]
