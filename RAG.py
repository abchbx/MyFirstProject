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

import os
import json
import random
from datetime import date
from llama_index.llms.openai_like import OpenAILike

os.environ['ZHIPU_API_KEY'] = '填写'
os.environ['Moonshot_API_KEY'] = '填写'
os.environ['ShuSHENG_API_KEY'] = '填写'

# 自定义异常类
class NoAvailableModelException(Exception):
    pass

# 模型信息列表，每个模型包含其API参数和每日最大使用次数
models_info = [
    {
        "model": "glm-4-flash",
        "api_base": "https://open.bigmodel.cn/api/paas/v4/",
        "api_key": os.getenv("ZHIPU_API_KEY"),
        "max_usage": 3  # 设置每天最大使用次数
    },
    {
        "model": "moonshot-v1-8k",
        "api_base": "https://api.moonshot.cn/v1",
        "api_key": os.getenv("Moonshot_API_KEY"),
        "max_usage": 3
    },
    {
        "model": "internlm2.5-latest",
        "api_base": "https://internlm-chat.intern-ai.org.cn/puyu/api/v1/",
        "api_key": os.getenv("ShuSHENG_API_KEY"),
        "max_usage": 3
    }
]



from datetime import datetime

# 定义文件名
USAGE_FILE = "model_usage.json"

# 初始化数据
initial_data = {
    "date": datetime.now().strftime('%Y-%m-%d'),  # 当前日期，格式为 YYYY-MM-DD
    "usage": {
        "moonshot-v1-8k": 0,
        "glm-4-flash": 0
    }
}

try:
    # 尝试打开文件并读取现有数据
    if os.path.exists(USAGE_FILE):
        with open(USAGE_FILE, 'r') as file:
            data = json.load(file)
    else:
        # 如果文件不存在，则初始化数据并写入文件
        data = initial_data
        with open(USAGE_FILE, 'w') as file:
            json.dump(data, file, indent=4)

except IOError as e:
    print(f"An error occurred while accessing the file: {e}")
except json.JSONDecodeError as e:
    print(f"An error occurred while parsing JSON: {e}")


def load_usage():
    """
    加载使用次数记录。如果记录的日期不是今天，则重置使用次数。
    """
    if os.path.exists(USAGE_FILE):
        with open(USAGE_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if data.get('date') == date.today().isoformat():
                return data.get('usage', {})
    # 如果没有文件或日期不是今天，返回空字典
    return {}

def save_usage(usage):
    """
    保存当前的使用次数记录到文件。
    """
    data = {
        'date': date.today().isoformat(),
        'usage': usage
    }
    with open(USAGE_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# 初始化使用次数
usage_counts = load_usage()

# 创建所有模型的实例，并将使用次数与模型关联
llm_instances = []
for info in models_info:
    model_name = info["model"]
    llm = OpenAILike(
        model=model_name,
        api_base=info["api_base"],
        api_key=info["api_key"],
        is_chat_model=True
    )
    llm_instances.append({
        'llm': llm,
        'info': info,
        'usage': usage_counts.get(model_name, 0)
    })

def get_available_models():
    """
    返回尚未达到使用上限的模型列表。
    """
    available = []
    for instance in llm_instances:
        if instance['usage'] < instance['info']['max_usage']:
            available.append(instance)
    return available

def get_random_model():
    """
    从可用的模型中随机选择一个，并更新其使用次数。
    如果所有模型都已达到使用上限，抛出自定义异常。
    """
    available_models = get_available_models()
    if not available_models:
        raise NoAvailableModelException("今日的使用次数已经用完，请明日再来")
    # 随机选择一个可用的模型
    model_instance = random.choice(available_models)
    # 更新使用次数
    model_instance['usage'] += 1
    usage_counts[model_instance['info']['model']] = model_instance['usage']
    save_usage(usage_counts)
    return model_instance['llm'], model_instance['info']['model']

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
persist_dir = "/mnt/workspace/llamaindex_RAG_onlineLLM/index" 
vector_store = FaissVectorStore.from_persist_dir(persist_dir)
storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=persist_dir)
index = load_index_from_storage(storage_context=storage_context, embed_model=embedding)

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

    try:
        # 获取随机模型及模型名称
        llm, model_name = get_random_model()

        # 构造流式输出引擎
        query_engine = index.as_query_engine(
            streaming=True,
            similarity_top_k=3,
            llm=llm,
        )

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

        # 将助手的完整回复添加到聊天记录中，包括模型名称
        assistant_response = f"{full_response}\n\n**模型名称:** {model_name}"
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})

    except NoAvailableModelException as e:
        with st.chat_message("assistant"):
            st.markdown(str(e))



