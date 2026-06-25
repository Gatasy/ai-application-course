# app.py
from qa_chain import MedExpertQA


def print_result(result):
    print(f"问题：{result['question']}")
    print(f"分类：{result['category']}")

    print("回答：")
    print(result["answer"])

    print("\n引用来源：")
    for src in result["sources"]:
        print(f"[{src['id']}] {src['source']} | score={src['score']:.4f}")
        print(f"    {src['preview']}")


def main():
    qa = MedExpertQA(top_k=3)

    print("MedExpert-RAG QA Chain")
    print("输入 exit 退出。\n")

    while True:
        question = input("请输入医学问题：").strip()

        if question.lower() in ["exit", "quit", "q"]:
            break

        if not question:
            continue

        try:
            result = qa.ask(question)
            print_result(result)
        except Exception as e:
            print(f"\n[Error] 问答链运行失败：{e}\n")


if __name__ == "__main__":
    main()