# retriever.py
import os
from pathlib import Path
from typing import List, Dict, Any

try:
    import torch
except ImportError:
    torch = None

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


BASE_DIR = Path(__file__).resolve().parent

# 对应你的实际目录：
# ai-application-development/rag/rag/vector_store/Disease
VECTOR_STORE_ROOT = BASE_DIR / "rag" / "rag" / "vector_store"

# 必须和 build_index.py 里保持一致
EMBEDDING_MODEL = "BAAI/bge-m3"


DOMAIN_MAP = {
    "Disease": "Disease",
    "Drug": "Drug",
    "Exam": "Exam",
    "disease": "Disease",
    "drug": "Drug",
    "exam": "Exam",
}


def get_device() -> str:
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


class MedRetriever:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": get_device()},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vector_stores = {}

    def normalize_domain(self, kb_type: str) -> str:
        if kb_type not in DOMAIN_MAP:
            raise ValueError(
                f"未知知识库类型：{kb_type}，只允许 Disease / Drug / Exam"
            )
        return DOMAIN_MAP[kb_type]

    def load_vector_store(self, kb_type: str):
        domain = self.normalize_domain(kb_type)

        if domain in self.vector_stores:
            return self.vector_stores[domain]

        store_path = VECTOR_STORE_ROOT / domain

        if not store_path.exists():
            raise FileNotFoundError(
                f"未找到向量库：{store_path}\n"
                f"请确认 build_index.py 已经运行，并且生成了 index.faiss 和 index.pkl。"
            )

        vector_store = FAISS.load_local(
            folder_path=str(store_path),
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True,
        )

        self.vector_stores[domain] = vector_store
        return vector_store

    def retrieve(
        self,
        question: str,
        kb_type: str,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        返回格式：
        [
            {
                "content": "...",
                "source": "...",
                "score": 0.123
            }
        ]
        """
        vector_store = self.load_vector_store(kb_type)

        docs_with_scores = vector_store.similarity_search_with_score(
            question,
            k=top_k
        )

        results = []
        for doc, score in docs_with_scores:
            results.append({
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "score": float(score),
            })

        return results


if __name__ == "__main__":
    retriever = MedRetriever()
    question = "What is aspirin used for?"
    results = retriever.retrieve(question, kb_type="Drug", top_k=3)
    for i, item in enumerate(results, start=1):
        print(f"\n[{i}] source={item['source']} score={item['score']}")
        print(item["content"][:500])