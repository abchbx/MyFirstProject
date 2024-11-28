import os
import faiss
from llama_index.core import (
    SimpleDirectoryReader,
    SummaryIndex,
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.embeddings import BaseEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.llms.openai_like import OpenAILike
from openai import OpenAI
from pydantic import Field
from typing import Any, List, Mapping, Optional
import streamlit as st

# 设置环境变量
os.environ['ZHIPU_API_KEY'] = ''
os.environ['ShuSheng_API_KEY'] = ''

# 初始化LLM和嵌入模型
llm = OpenAILike(
    model="glm-4-flash",
    api_base="https://open.bigmodel.cn/api/paas/v4/",
    api_key=os.getenv("ZHIPU_API_KEY"),
    is_chat_model=True,
)

class ZhipuEmbeddings(BaseEmbedding):
    client: OpenAI = Field(
        default_factory=lambda: OpenAI(
            api_key=os.environ['ZHIPU_API_KEY'],
            base_url="https://open.bigmodel.cn/api/paas/v4/"
        )
    )

    def __init__(self, model_name: str = "embedding-2", **kwargs: Any) -> None:
        super().__init__(model_name=model_name, **kwargs)
        self._model = model_name

    def invoke_embedding(self, query: str) -> List[float]:
        response = self.client.embeddings.create(model=self._model, input=[query])
        if response.data and len(response.data) > 0:
            return response.data[0].embedding
        else:
            raise ValueError("Failed to get embedding from ZhipuAI API")

    def _get_query_embedding(self, query: str) -> List[float]:
        return self.invoke_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self.invoke_embedding(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return [self._get_text_embedding(text) for text in texts]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._get_text_embeddings(texts)

# 创建嵌入模型实例
embedding = ZhipuEmbeddings()

# 设置持久化目录和向量存储
persist_dir = "/mnt/workspace/article/index" 
vector_store = FaissVectorStore.from_persist_dir(persist_dir)
storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=persist_dir)
index = load_index_from_storage(storage_context=storage_context, embed_model=embedding)

# 构造流式输出引擎
query_engine = index.as_query_engine(
    streaming=True,
    similarity_top_k=3,
    llm=llm,
)

import streamlit as st
import json

st.title("KnowYourself")

# 初始化聊天记录
if "messages" not in st.session_state:
    st.session_state.messages = []
# 添加侧边栏
with st.sidebar:
    # 添加一个按钮来开始新的对话
    if st.button("New Conversation"):
        st.session_state.messages = []
# 显示聊天记录中的消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 接收用户输入
if user_input := st.chat_input("请输入您的问题:"):
    # 将用户消息添加到聊天记录中
    st.session_state.messages.append({"role": "user", "content": user_input})
    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(user_input)

    # 创建一个空容器用于显示助手的回复
    with st.chat_message("assistant"):
        response_container = st.empty()

    # 查询并获取回复流
    def response_generator():
        response_stream = query_engine.query(user_input)

        if hasattr(response_stream, 'response_gen'):
            for chunk in response_stream.response_gen:
                yield chunk
        else:
            st.error("Response stream does not have a recognized attribute for streaming.")
            return

    # 流式显示助手的回复
    full_response = ""
    for chunk in response_generator():
        full_response += chunk
        response_container.markdown(full_response)

    # 将助手的完整回复添加到聊天记录中
    st.session_state.messages.append({"role": "assistant", "content": full_response})