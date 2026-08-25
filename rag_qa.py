import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI

# 初始化 DeepSeek 客户端（与之前相同）
client = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com")

# 加载已保存的向量数据库
embeddings = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")
vectorstore = Chroma(persist_directory="./chroma_db",embedding_function=embeddings)


def ask_rag(question):
    # 1. 从向量数据库中检索最相关的 3 个文本段
    docs = vectorstore.similarity_search(question, k=3)
    # 2. 把检索到的文本拼接成上下文
    context = "\n\n".join([doc.page_content for doc in docs])

    # 3. 构造 Prompt
    prompt = f"""你是一个资深的博物馆文物专家。请根据以下参考资料回答用户的问题。
如果参考资料中没有相关信息，请如实告知，不要编造。

参考资料：
{context}

用户问题：{question}

你的回答："""

    # 4. 调用 DeepSeek 生成回答
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一个严谨的博物馆文物专家。"},
            {"role": "user", "content": prompt}
        ],
        stream=False
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    # 测试几个问题
    test_questions = [
        "介绍一下你资料中的青铜器",
        "有哪些关于贵州少数民族的文物？",
        "博物馆里有什么？"
    ]
    for q in test_questions:
        print(f"问题：{q}")
        print(f"回答：{ask_rag(q)}\n")
        print("-" * 50)