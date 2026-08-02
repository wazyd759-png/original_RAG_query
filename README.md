#RAG 问答助手 —— 基于 LangChain 的本地知识库检索增强生成

本项目是一个简单的 RAG（Retrieval-Augmented Generation）示例，使用阿里云百炼平台（DashScope）的嵌入模型和通义千问大模型，配合 Chroma 向量数据库，实现从本地文本文件中检索相关片段，并让大模型基于这些片段回答用户问题。

##功能特点
- 加载本地 `.txt` 文件作为知识源
- 使用 `text-embedding-v2` 将文本切片为向量并存储
- 支持余弦相似度检索（`cosine`）
- 基于检索结果调用 `qwen-turbo` 生成回答
- 完全可配置，便于替换模型、数据或向量库

##安装与配置

###环境要求
- Python 3.8+
- 阿里云 DashScope API Key（[申请地址](https://dashscope.aliyun.com/)）

###1. 克隆项目
```bash
git clone <your-repo-url>
cd <project-folder>

## 📸 功能截图

![运行效果截图](./images/demo_01.png)
