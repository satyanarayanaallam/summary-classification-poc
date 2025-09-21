"""Simple FAISS-backed in-memory vector store for POC.

This store uses scikit-learn's TfidfVectorizer to build dense vectors for
triplet text and stores them in a FAISS index. It's intentionally simple and
meant for local POC runs without calling external embedding APIs.
"""
from typing import List, Dict, Any, Optional
import numpy as np

try:
    import faiss
except Exception:  # pragma: no cover - import may fail in CI if not installed
    faiss = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception:  # pragma: no cover - optional dependency
    TfidfVectorizer = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None


class FaissVectorStore:
    def __init__(self, dim: int = 384, model_name: str = "all-MiniLM-L6-v2"):
        self._texts: List[str] = []
        self._metas: List[Dict[str, Any]] = []
        self._dim = dim
        self._index = None
        self._model_name = model_name
        
        # Use Sentence Transformers as primary embedding method
        if SentenceTransformer is not None:
            self._embedder = SentenceTransformer(model_name)
            self._use_embedder = True
            # Update dim to match the actual model dimension
            actual_dim = self._embedder.get_sentence_embedding_dimension()
            self._dim = actual_dim
            print(f"Using Sentence Transformers model '{model_name}' with dimension {actual_dim}")
        else:
            if TfidfVectorizer is not None:
                print("Warning: sentence-transformers not available, falling back to TF-IDF")
                self._vectorizer = TfidfVectorizer(max_features=dim)
                self._use_embedder = False
            else:
                print("Warning: Neither sentence-transformers nor sklearn available. Using dummy embeddings.")
                self._vectorizer = None
                self._use_embedder = False

    def _ensure_index(self, vectors: np.ndarray):
        if faiss is None:
            raise RuntimeError("faiss is not installed; please install faiss-cpu")
        if self._index is None:
            self._index = faiss.IndexFlatIP(self._dim)
        # faiss expects float32
        self._index.add(vectors.astype('float32'))

    def add(self, texts: List[str], metas: List[Dict[str, Any]]):
        # extend texts & metas
        self._texts.extend(texts)
        self._metas.extend(metas)
        # compute embeddings
        if self._use_embedder:
            X = self._embedder.encode(self._texts, show_progress_bar=False)
            X = np.asarray(X)
            # if embedder returns vectors not matching dim, adjust dim
            if X.shape[1] != self._dim:
                if X.shape[1] < self._dim:
                    X = np.pad(X, ((0, 0), (0, self._dim - X.shape[1])), 'constant')
                else:
                    X = X[:, : self._dim]
        else:
            if self._vectorizer is not None:
                # re-fit vectorizer on all texts (simple POC approach)
                X = self._vectorizer.fit_transform(self._texts).toarray()
                # convert to fixed dim
                if X.shape[1] < self._dim:
                    X = np.pad(X, ((0, 0), (0, self._dim - X.shape[1])), 'constant')
                elif X.shape[1] > self._dim:
                    X = X[:, : self._dim]
            else:
                # Create dummy embeddings when no embedding method is available
                X = np.random.random((len(self._texts), self._dim)).astype(np.float32)
        self._index = None
        self._ensure_index(X)

    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if self._index is None:
            return []
        if self._use_embedder:
            q = self._embedder.encode([query_text], show_progress_bar=False)
            q = np.asarray(q)
            if q.shape[1] < self._dim:
                q = np.pad(q, ((0, 0), (0, self._dim - q.shape[1])), 'constant')
            elif q.shape[1] > self._dim:
                q = q[:, : self._dim]
        else:
            if self._vectorizer is not None:
                q = self._vectorizer.transform([query_text]).toarray()
                if q.shape[1] < self._dim:
                    q = np.pad(q, ((0, 0), (0, self._dim - q.shape[1])), 'constant')
                elif q.shape[1] > self._dim:
                    q = q[:, : self._dim]
            else:
                # Create dummy query embedding when no embedding method is available
                q = np.random.random((1, self._dim)).astype(np.float32)
        q = q.astype('float32')
        D, I = self._index.search(q, top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self._metas):
                continue
            meta = self._metas[idx].copy()
            meta["score"] = float(score)
            results.append(meta)
        return results
