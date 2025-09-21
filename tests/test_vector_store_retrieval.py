from src.db.vector_store import FaissVectorStore
from src.services.retrieval_service import RetrievalService


def test_index_and_query():
    store = FaissVectorStore(dim=32)
    service = RetrievalService(store)
    triplets = [("invoice", "has_amount", "<AMOUNT>"), ("invoice", "issued_by", "organization")]
    metas = [{"doc_id": "D1", "doc_type": "INVOICE", "doc_code": "INV001"},
             {"doc_id": "D1", "doc_type": "INVOICE", "doc_code": "INV001"}]
    service.index_triplets(triplets, metas)
    res = service.retrieve_by_triplet(("invoice", "issued_by", "organization"), top_k=2)
    assert res["doc_type"] == "INVOICE"
    assert res["doc_code"] == "INV001"
