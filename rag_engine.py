"""RAG 核心引擎：多 markdown 文档加载 / 切分 / 检索 / 重排 / 生成 / 增量入库。"""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.llms import Tongyi
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from sentence_transformers import CrossEncoder

PROMPT = """你是一个专业的求职知识库助手。请根据以下提供的上下文信息回答用户的问题。

重要规则：
1. 优先参考与问题主体最匹配的上下文。例如，用户问"上海银行"，就优先使用包含"上海银行"的上下文。
2. 如果某段上下文的主体（公司、岗位、人名等）与问题主体明显不一致，请忽略该段，不要用它来回答。
3. 如果所有上下文都不包含答案，请如实告知"知识库中没有找到相关信息"，不要编造。

上下文：
{context}

用户问题：
{question}

请给出简洁、准确的回答：
"""

# 文件名匹配时给的加成（加到重排分上）
FILENAME_BONUS = 1.5


class RAGEngine:
    def __init__(
        self,
        data_dir: str = "/app/data",
        persist_dir: str = "/app/chroma_db1",
        collection_name: str = "demo",
        rebuild: bool = False,
        retrieve_k: int = 7,
        rerank_top_k: int = 3,
        embedding_model: str = "text-embedding-v2",
        rerank_model: str = "BAAI/bge-reranker-base",
        llm_model: str = "qwen-turbo",
    ) -> None:
        self.data_dir = data_dir
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.retrieve_k = retrieve_k
        self.rerank_top_k = rerank_top_k

        self.embeddings = DashScopeEmbeddings(
            model=embedding_model,
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
        )

        if rebuild and os.path.isdir(persist_dir):
            for name in os.listdir(persist_dir):
                p = os.path.join(persist_dir, name)
                if os.path.isdir(p):
                    shutil.rmtree(p)
                else:
                    os.remove(p)
            print(f"[RAG] 已清空旧向量库内容: {persist_dir}")

        self.db = self._build_or_load()

        self.retriever = self.db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.retrieve_k},
        )

        print("[RAG] 正在加载重排模型 ...")
        self.reranker = CrossEncoder(rerank_model)

        print("[RAG] 正在加载 LLM ...")
        self.llm = Tongyi(model=llm_model)

        self.prompt_template = ChatPromptTemplate.from_messages([("human", PROMPT)])
        print("[RAG] 引擎就绪 ✅")

    # ---------- 加载目录下所有 markdown ----------
    def _load_markdown_docs(self) -> List[Document]:
        data_path = Path(self.data_dir)
        md_files = sorted(data_path.rglob("*.md"))
        if not md_files:
            raise FileNotFoundError(f"在 {self.data_dir} 下没找到任何 .md 文件")

        print(f"[RAG] 发现 {len(md_files)} 个 markdown 文件")

        md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
            strip_headers=False,
        )
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "。", "？", "！", "；", "，", "、", "\n", " ", ""],
        )

        all_chunks: List[Document] = []
        for md_file in md_files:
            content = md_file.read_text(encoding="utf-8")
            header_chunks = md_splitter.split_text(content)

            for chunk in header_chunks:
                chunk.metadata["source_file"] = md_file.name
                chunk.metadata["source_path"] = str(md_file.relative_to(data_path))
                sub_chunks = text_splitter.split_documents([chunk])
                all_chunks.extend(sub_chunks)

        print(f"[RAG] 切分后共 {len(all_chunks)} 个块")
        return all_chunks

    # ---------- 建库 / 加载 ----------
    def _build_or_load(self) -> Chroma:
        if os.path.isdir(self.persist_dir) and os.listdir(self.persist_dir):
            print(f"[RAG] 加载已有向量库: {self.persist_dir}")
            return Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_dir,
            )

        print(f"[RAG] 构建向量库，数据源目录: {self.data_dir}")
        documents = self._load_markdown_docs()

        return Chroma.from_documents(
            collection_name=self.collection_name,
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_dir,
            collection_metadata={"hnsw:space": "cosine"},
        )

    # ---------- 增量入库单个文件 ----------
    def add_file(self, file_path: str) -> Dict[str, Any]:
        """把一份 md/txt 文件切分、embedding、增量写入向量库。"""
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        ext = p.suffix.lower()
        if ext not in (".md", ".txt"):
            raise ValueError(f"不支持的文件类型: {ext}（仅支持 .md / .txt）")

        content = p.read_text(encoding="utf-8", errors="ignore")
        if not content.strip():
            raise ValueError("文件内容为空")

        if ext == ".md":
            md_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
                strip_headers=False,
            )
            header_chunks = md_splitter.split_text(content)
        else:
            header_chunks = [Document(page_content=content)]

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "。", "？", "！", "；", "，", "、", "\n", " ", ""],
        )

        all_chunks: List[Document] = []
        for chunk in header_chunks:
            chunk.metadata["source_file"] = p.name
            chunk.metadata["source_path"] = f"uploads/{p.name}"
            sub_chunks = text_splitter.split_documents([chunk])
            all_chunks.extend(sub_chunks)

        self.db.add_documents(all_chunks)
        total = self.db._collection.count()
        print(f"[RAG] 已入库 {p.name}，新增 {len(all_chunks)} 块，总计 {total} 块")

        return {"source_file": p.name, "chunks": len(all_chunks), "total": total}

    # ---------- 向量库统计 ----------
    def stats(self) -> Dict[str, Any]:
        return {"total_chunks": self.db._collection.count()}

    # ---------- 文件名加成 ----------
    @staticmethod
    def _filename_bonus(question: str, filename: str) -> float:
        """文件名按分隔符切片，任意一片出现在问题里就加成。"""
        if not filename:
            return 0.0
        stem = filename.rsplit(".", 1)[0]  # 去掉扩展名
        parts = re.split(r"[-_·\s/（）()【】\[\]]+", stem)
        for part in parts:
            part = part.strip()
            if len(part) >= 2 and part in question:
                return FILENAME_BONUS
        return 0.0

    # ---------- 重排 + 文件名加成 ----------
    def _rank(
        self, question: str, docs: List[Document]
    ) -> List[Tuple[Document, float, float]]:
        """返回 [(doc, 原始重排分, 加成后分)]，按加成后分从高到低排序。"""
        pairs = [[question, d.page_content] for d in docs]
        scores = self.reranker.predict(pairs)

        results: List[Tuple[Document, float, float]] = []
        for doc, score in zip(docs, scores):
            raw = float(score)
            fname = doc.metadata.get("source_file", "")
            bonus = self._filename_bonus(question, fname)
            results.append((doc, raw, raw + bonus))

        results.sort(key=lambda x: x[2], reverse=True)
        return results

    # ---------- 问答 ----------
    def ask(self, question: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        question = (question or "").strip()
        if not question:
            raise ValueError("问题不能为空")

        retrieved_docs = self.retriever.invoke(question)
        if not retrieved_docs:
            return {"answer": "抱歉，知识库中没有检索到相关内容。", "sources": []}

        ranked = self._rank(question, retrieved_docs)
        k = top_k or self.rerank_top_k
        top = ranked[:k]

        context_text = "\n\n---\n\n".join(d.page_content for d, _, _ in top)
        formatted_prompt = self.prompt_template.format(
            question=question, context=context_text
        )
        answer = self.llm.invoke(formatted_prompt)

        sources: List[Dict[str, Any]] = [
            {
                "index": i + 1,
                "content": doc.page_content,
                "score": round(raw, 4),
                "final_score": round(final, 4),
                "source_file": doc.metadata.get("source_file", ""),
                "source_path": doc.metadata.get("source_path", ""),
                "metadata": dict(doc.metadata or {}),
            }
            for i, (doc, raw, final) in enumerate(top)
        ]

        return {"answer": answer, "sources": sources}