import streamlit as st
import os
from openai import OpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# 加载已保存的向量数据库
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    return vectorstore

vectorstore = load_vectorstore()

def ask_rag(question):
    docs = vectorstore.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"""你是一个资深的博物馆文物专家。请根据以下参考资料回答用户的问题。
如果参考资料中没有相关信息，请如实告知，不要编造。

参考资料：
{context}

用户问题：{question}

你的回答："""
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一个严谨的博物馆文物专家。"},
            {"role": "user", "content": prompt}
        ],
        stream=False
    )
    return response.choices[0].message.content
# 设置页面
st.set_page_config(
    page_title="博物馆文物知识问答",
    page_icon="🏛️",
    layout="wide"
)

# 初始化 OpenAI 客户端（DeepSeek 兼容 OpenAI 接口）

client = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com")

st.title('🏛️ 博物馆文物知识问答')

# 初始化会话状态，用于保存聊天记录
if 'messages' not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.write(message["content"])
    elif message["role"] == "assistant":
        with st.chat_message("assistant"):
            st.write(message["content"])

# 获取用户输入
question = st.chat_input("请输入你想要查询的文物问题，例如：什么是青铜器？")
if question:
    # 显示用户消息并保存
    with st.chat_message("user"):
        st.write(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # 调用 DeepSeek API
    try:
        assistant_content = ask_rag(question)
    except Exception as e:
        assistant_content = f"调用出错：{e}"

    with st.chat_message("assistant"):
        st.write(assistant_content)
    st.session_state.messages.append({"role": "assistant", "content": assistant_content})
