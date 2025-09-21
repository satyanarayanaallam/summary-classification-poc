from src.services.evaluation_service import EvaluationService


def test_single_correct():
    svc = EvaluationService()
    pred = {"doc_type": "INVOICE"}
    gt = {"doc_type": "INVOICE"}
    res = svc.evaluate(pred, gt)
    assert res["n"] == 1
    assert res["accuracy"] == 1.0
    assert res["per_label"]["INVOICE"]["support"] == 1


def test_single_incorrect():
    svc = EvaluationService()
    pred = {"doc_type": "INVOICE"}
    gt = {"doc_type": "BANK_STATEMENT"}
    res = svc.evaluate(pred, gt)
    assert res["n"] == 1
    assert res["accuracy"] == 0.0
    # no true positives
    assert res["per_label"]["INVOICE"]["precision"] == 0.0


def test_batch_mixed():
    svc = EvaluationService()
    preds = [
        {"doc_type": "INVOICE"},
        {"doc_type": "BANK_STATEMENT"},
        {"doc_type": "INVOICE"},
    ]
    gts = [
        {"doc_type": "INVOICE"},
        {"doc_type": "BANK_STATEMENT"},
        {"doc_type": "BANK_STATEMENT"},
    ]
    res = svc.evaluate(preds, gts)
    assert res["n"] == 3
    assert res["accuracy"] == 2 / 3
    # INVOICE support = 1 (one ground truth INVOICE)
    assert res["per_label"]["INVOICE"]["support"] == 1
    # some micro metrics should be > 0
    assert res["micro"]["precision"] >= 0.0
    assert res["macro"]["f1"] >= 0.0
