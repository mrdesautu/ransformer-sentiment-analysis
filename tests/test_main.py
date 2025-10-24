import pytest

from src import main


class DummyPipeline:
    def __call__(self, text):
        return [{"label": "POSITIVE", "score": 0.99, "text": text}]


def test_predict_happy_path(monkeypatch):
    # Mock the transformers.pipeline constructor
    monkeypatch.setattr(main, "pipeline", lambda task, model=None: DummyPipeline())

    out = main.predict("Hello world", model_name="dummy-model", task="sentiment-analysis")
    assert out["text"] == "Hello world"
    assert out["model"] == "dummy-model"
    assert out["task"] == "sentiment-analysis"
    assert isinstance(out["result"], list)


def test_predict_type_error():
    with pytest.raises(TypeError):
        main.predict(123)  # type: ignore
