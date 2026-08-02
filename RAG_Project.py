from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Tongyi
import os
import shutil

# ======================== 1. 加载与切分 ========================
loader = TextLoader(file_path=r"D:\demo1\data\产品1.txt", encoding="utf-8")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=[
        "\n\n", "。", "？", "！", "；", "，", "、", "\n", " ", ""
    ]
)
documents = text_splitter.split_documents(docs)
print(f"✅ 切分后共有 {len(documents)} 个块")

#强制重建向量数据库
persist_dir = "./chroma_db1"
if os.path.exists(persist_dir):
    shutil.rmtree(persist_dir)
    print(f"已删除旧的向量库: {persist_dir}")

embeddings = DashScopeEmbeddings(
    model="text-embedding-v2",
    dashscope_api_key=os.getenv('DASHSCOPE_API_KEY')
)

db = Chroma.from_documents(
    collection_name='demo',
    documents=documents,
    embedding=embeddings,
    persist_directory=persist_dir,
    collection_metadata={"hnsw:space": 'cosine'}
)


#检索
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 7}   # 总共7个块，全部取回
)

test_query = "GPT5.6系列产品如何被推出的" 
retrieved_docs = retriever.invoke(test_query)

# ---------- 关键步骤：手动拼接上下文 ----------
context_text = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])


 
message = """
你是一个专业的问答助手。请根据以下提供的上下文信息，回答用户的问题。
如果上下文不包含答案，请如实告知。

上下文：
{context}

用户问题：
{question}

请给出简洁、准确的回答：
"""
prompt_template = ChatPromptTemplate.from_messages([("human", message)])

# 直接格式化提示词
formatted_prompt = prompt_template.format(question=test_query, context=context_text)


llm = Tongyi(model="qwen-turbo")

# 直接调用 LLM
response = llm.invoke(formatted_prompt)

print("="*50)
print(response)