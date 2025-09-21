from src.utils.normalization import normalize_triplet, triplet_to_text


def test_normalize_amount_date():
    t = ("AMOUNT", "has_value", "$1,200.00")
    n = normalize_triplet(t)
    assert n[0] == "amount" or n[0] == "AMOUNT".lower()
    assert n[1] == "has_amount"
    assert "<AMOUNT>" in n[2]


def test_triplet_to_text():
    t = ("Invoice", "issued", "ACME Corp")
    txt = triplet_to_text(t)
    assert txt.startswith("invoice issued_by")
