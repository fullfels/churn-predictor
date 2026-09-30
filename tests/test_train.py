from train import build_pipeline, make_customer_data, train


def test_generated_data_has_expected_shape():
    data = make_customer_data(200)
    assert data.shape == (200, 8)
    assert 0 < data["churn"].mean() < 1


def test_pipeline_trains_with_useful_signal():
    model, metrics, _, _ = train()
    assert hasattr(model, "predict_proba")
    assert metrics["roc_auc"] > 0.75
    assert build_pipeline() is not None
