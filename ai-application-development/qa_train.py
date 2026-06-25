# qa_chain.py
import json
import os
import re
from pathlib import Path
from typing import Dict, Any, List
from config import API_KEY, BASE_URL, MODEL_NAME
from retriever import MedRetriever


BASE_DIR = Path(__file__).resolve().parent
PROMPT_DIR = BASE_DIR / "prompts"


def load_prompt(prompt_type: str) -> str:
    """
    prompt_type: router / disease / drug / exam
    """
    prompt_type = prompt_type.lower()
    prompt_path = PROMPT_DIR / f"{prompt_type}.txt"

    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt 文件不存在：{prompt_path}")

    return prompt_path.read_text(encoding="utf-8")


def render_prompt(template: str, **kwargs) -> str:
    """
    不用 str.format，避免 prompt 里有 JSON 大括号时报错。
    只做简单占位符替换。
    """
    prompt = template
    for key, value in kwargs.items():
        prompt = prompt.replace("{" + key + "}", str(value))
    return prompt

def call_llm(prompt: str) -> str:
    """
    使用 config.py 中的 API_KEY / BASE_URL / MODEL_NAME 调用 OpenAI-compatible API。
    """
    try:
        from config import API_KEY, BASE_URL, MODEL_NAME
    except ImportError:
        raise RuntimeError(
            "没有找到 config.py。请在项目根目录创建 config.py，并设置 API_KEY、BASE_URL、MODEL_NAME。"
        )

    if not API_KEY:
        raise RuntimeError("config.py 中的 API_KEY 为空，请检查配置。")

    from openai import OpenAI

    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful medical knowledge assistant. "
                    "Answer only based on the retrieved medical knowledge context."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()

def parse_category(router_output: str) -> str:
    """
    Router 期望输出：
    {"category": "Disease"}
    或 {"category": "Drug"}
    或 {"category": "Exam"}
    """
    text = router_output.strip()

    try:
        data = json.loads(text)
        category = data.get("category", "")
    except Exception:
        match = re.search(r"Disease|Drug|Exam", text, re.IGNORECASE)
        category = match.group(0) if match else ""

    category = category.lower()

    if category == "disease":
        return "Disease"
    if category == "drug":
        return "Drug"
    if category == "exam":
        return "Exam"

    raise ValueError(f"Router 输出无法解析为合法分类：{router_output}")


def rule_based_router(question: str) -> str:
    """
    兜底规则：防止 Router API 暂时不可用时整个程序没法调试。
    最终验收最好还是用 LLM Router。
    """
    q = question.lower()

    drug_words = [
        "drug", "medicine", "medication", "tablet", "capsule",
        "aspirin", "metformin", "insulin", "amlodipine",
        "atorvastatin", "omeprazole", "clopidogrel"
    ]

    exam_words = [
        "test", "exam", "scan", "x-ray", "ct", "ecg",
        "blood glucose", "blood count", "urinalysis",
        "liver function", "renal function", "thyroid"
    ]

    if any(w in q for w in drug_words):
        return "Drug"
    if any(w in q for w in exam_words):
        return "Exam"
    return "Disease"


def build_context(retrieved_docs: List[Dict[str, Any]]) -> str:
    context_parts = []

    for i, item in enumerate(retrieved_docs, start=1):
        content = item["content"]
        source = item["source"]
        context_parts.append(f"[{i}] Source: {source}\n{content}")

    return "\n\n".join(context_parts)


def format_sources(retrieved_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sources = []

    for i, item in enumerate(retrieved_docs, start=1):
        sources.append({
            "id": i,
            "source": item["source"],
            "score": item["score"],
            "preview": item["content"][:120].replace("\n", " ") + "..."
        })

    return sources


class MedExpertQA:
    def __init__(self, top_k: int = 3):
        self.top_k = top_k
        self.retriever = MedRetriever()

    def route_question(self, question: str) -> str:
        router_template = load_prompt("router")

        router_input = json.dumps(
            {"question": question},
            ensure_ascii=False
        )

        router_prompt = render_prompt(
            router_template,
            question=question,
            input=router_input,
        )

        try:
            router_output = call_llm(router_prompt)
            return parse_category(router_output)
        except Exception as e:
            print(f"[Warning] Router 调用失败，使用规则兜底分类。原因：{e}")
            return rule_based_router(question)

    def ask(self, question: str) -> Dict[str, Any]:
        category = self.route_question(question)

        retrieved_docs = self.retriever.retrieve(
            question=question,
            kb_type=category,
            top_k=self.top_k,
        )

        context = build_context(retrieved_docs)

        expert_prompt_type = category.lower()
        expert_template = load_prompt(expert_prompt_type)

        qa_prompt = render_prompt(
            expert_template,
            context=context,
            question=question,
        )

        answer = call_llm(qa_prompt)

        return {
            "question": question,
            "category": category,
            "answer": answer,
            "sources": format_sources(retrieved_docs),
        }