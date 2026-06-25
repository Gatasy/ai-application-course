# retriever.py
import os
from pathlib import Path
from typing import List, Dict, Any
import re

try:
    import torch
except ImportError:
    torch = None

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


BASE_DIR = Path(__file__).resolve().parent

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

    def normalize_text(self, text: str) -> str:
        """
        统一大小写、下划线、连字符和多余空格，方便实体名匹配。
        例如：
        Diabetes_Mellitus -> diabetes mellitus
        Blood_Glucose_Test -> blood glucose test
        """
        text = text.lower()
        text = text.replace("_", " ")
        text = text.replace("-", " ")
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text


    def get_all_documents(self, vector_store):
        """
        从 FAISS 的 docstore 里取出所有文档。
        这个适合你们这种小型 KB，可以做实体名精确匹配。
        """
        docs = []

        for doc_id in vector_store.index_to_docstore_id.values():
            doc = vector_store.docstore.search(doc_id)
            docs.append(doc)

        return docs


    def retrieve(
        self,
        question: str,
        kb_type: str,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        混合检索逻辑：
        1. 先根据 source 实体名做精确匹配
        2. 如果问题中包含实体名，优先返回该实体相关 chunk
        3. 再用 FAISS 向量检索补足 top_k
        """
        vector_store = self.load_vector_store(kb_type)

        question_norm = self.normalize_text(question)

        # 1. 先从整个 docstore 里做实体名匹配
        all_docs = self.get_all_documents(vector_store)

        exact_results = []
        
        for doc in all_docs:
            source = doc.metadata.get("source", "unknown")
            source_norm = self.normalize_text(source)

            # 关键：如果问题中包含 source 名，例如 diabetes mellitus / aspirin / blood glucose test
            if source_norm and source_norm in question_norm:
                exact_results.append({
                    "content": doc.page_content,
                    "source": source,
                    "score": 0.0,
                    "match_type": "entity_match",
                })
        if exact_results:
            return exact_results[:top_k]

        # 2. 再跑向量检索，用来补足
        docs_with_scores = vector_store.similarity_search_with_score(
            question,
            k=max(top_k * 3, 10)
        )

        vector_results = []
        for doc, score in docs_with_scores:
            source = doc.metadata.get("source", "unknown")

            vector_results.append({
                "content": doc.page_content,
                "source": source,
                "score": float(score),
                "match_type": "vector_search",
            })

        # 3. 合并去重：同一个 source + content 只保留一次
        merged = []
        seen = set()

        for item in exact_results + vector_results:
            key = (item["source"], item["content"][:100])
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)

        return merged[:top_k]


if __name__ == "__main__":
    retriever = MedRetriever()

    test_questions = [
        "What are the symptoms of hypertension?",
        "How is bronchial asthma treated?",
        "What causes chronic gastritis?",
        "What are the prevention advice for coronary heart disease?",
        "What are the symptoms of Parkinsons disease?",
        "What is COPD?",
        "How to prevent rheumatoid arthritis?",
        "What are the treatment principles of cerebral infarction?",
    ]

    for question in test_questions:
        print("\n==============================")
        print("Question:", question)

        results = retriever.retrieve(question, kb_type="Disease", top_k=3)

        for i, item in enumerate(results, start=1):
            print(
                f"\n[{i}] source={item['source']} "
                f"score={item['score']} "
                f"match_type={item.get('match_type')}"
            )
            print(item["content"][:300])