# KnowYourself_RAG
## 概述
本项目基于[wow-rag](https://github.com/datawhalechina/wow-rag/tree/main/docs) 教程搭建,作为个人的学习项目。
### 使用方法
1. 配置环境变量
   a. 可以直接在文件中配置（不建议）
   b. export MY_VAR="my_value" (直接在终端运行，只对当前终端生效)
   c. 在 ~/.bashrc下配置（启动服务器即生效）
2. 安装依赖项
```
  pip install Streamlit
  pip install faiss-cpu scikit-learn scipy 
  pip install openai
  pip install llama-index-core 
  pip install llama-index-llms-openai-like 
  pip install llama-index-readers-file 
  pip install llama-index-vector-stores-faiss 
  pip install llamaindex-py-client
```
4. 启动
`streamlit run RAG.py`
### 工作流程（大纲）
1. 爬取公众号的文章，作为RAG知识库
2. 用data-juicer对数据进行一次简单清洗
3. 用llamaindex对知识库进行分块并建立索引
4. 用llamaindex串联知识库和大模型
5. 用streamlit创建前端应用
#### 其他待解决问题
1. (向量数据库)[https://qdrant.tech/]
2. (RAG的评估) [https://github.com/explodinggradients/ragas]
3. (工具的调用)[https://www.aidoczh.com/llamaindex/module_guides/deploying/agents/tools/]
4. (长期记忆)[https://github.com/mem0ai/mem0]

