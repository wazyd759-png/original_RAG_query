from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Tongyi
import os

# 加载和切分文档（略）
loader = TextLoader(file_path=r"D:\documents合集\产品1.txt", encoding="utf-8")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
documents = text_splitter.split_documents(docs)

# 嵌入模型
embeddings = DashScopeEmbeddings(
    model="text-embedding-v2",
    dashscope_api_key=os.getenv('DASHSCOPE_API_KEY')
)

# 向量库（注意参数名）
db = Chroma.from_documents(
    collection_name='demo',
    documents=documents,
    embedding=embeddings,                     # ✅ 修正
    persist_directory="./chroma_db1",        # 可选
    collection_metadata={"hnsw:space": 'cosine'}  # 若需余弦距离
)

# 检索器
docs_find = RunnableLambda(db.similarity_search).bind(k=2)

# Prompt 模板
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

llm = Tongyi(model="qwen-turbo")

# 构建 Chain
chain = {
    "question": RunnablePassthrough(),
    "context": docs_find
} | prompt_template | llm

response = chain.invoke("GPT-5.6有哪三种型号")
print(response)