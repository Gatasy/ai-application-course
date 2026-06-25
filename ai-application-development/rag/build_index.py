import os
import json
from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_ROOT = "data"
VECTOR_STORE_ROOT = "rag/vector_store"

DOMAINS = {
    "Disease": "Disease_KB",
    "Drug": "Drug_KB",
    "Exam": "Exam_KB"
}

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
)

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True}
)

def build_vector_store(domain_name, folder_path):
    print(f"正在处理领域: {domain_name}")
    all_documents = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            print(f"  加载文件: {filename}")
            try:
                loader = JSONLoader(
                    file_path=file_path,
                    jq_schema='.',  
                    text_content=False,
                    json_lines=False
                )
                documents = loader.load()
                
                for doc in documents:
                    doc.metadata["source"] = filename.replace('.json', '')
                    data = json.load(open(file_path, 'r', encoding='utf-8'))
                    doc.page_content = f"""
# {data.get('name', '')}

## Definition
{data.get('definition', '')}

## Etiology
{data.get('etiology', '')}

## Symptoms
{data.get('symptoms', '')}

## Treatment Principles
{data.get('treatment_principles', '')}

## Prevention Advice
{data.get('prevention_advice', '')}

## Source
{data.get('source', 'CMeKG/公开医学资料')}
"""
                all_documents.extend(documents)
            except Exception as e:
                print(f"  加载文件 {filename} 失败: {e}")

    if not all_documents:
        print(f"警告: 领域 {domain_name} 没有找到任何JSON文件。")
        return

    print(f"  正在切分文档...")
    chunked_documents = text_splitter.split_documents(all_documents)
    print(f"  切分完成，共生成 {len(chunked_documents)} 个文本块。")

    print(f"  正在构建向量索引...")
    vector_store = FAISS.from_documents(chunked_documents, embeddings)
    
    save_path = os.path.join(VECTOR_STORE_ROOT, domain_name)
    vector_store.save_local(save_path)
    print(f"  向量索引已保存至: {save_path}\n")

def main():
    os.makedirs(VECTOR_STORE_ROOT, exist_ok=True)
    for domain_name, folder_name in DOMAINS.items():
        folder_path = os.path.join(DATA_ROOT, folder_name)
        if os.path.exists(folder_path):
            build_vector_store(domain_name, folder_path)
        else:
            print(f"警告: 文件夹 {folder_path} 不存在，跳过领域 {domain_name}。")
    print("所有向量索引构建完成！")

if __name__ == "__main__":
    main()