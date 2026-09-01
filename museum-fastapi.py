import streamlit as st
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import uvicorn
from openai import OpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="博物馆文物知识问答系统")
embeddings = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# 创建 OpenAI 客户端
client = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com"
)

class RAG:
    def __init__(self, vectorstore,client):
        self.vectorstore = vectorstore
        self.client = client

    def ask_rag(self,question):
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

rag_service = RAG(vectorstore, client)

class QuestionRequest(BaseModel):
    question: str

# ---------- API 端点 ----------
@app.post("/ask")
async def ask_question(request: QuestionRequest):
    answer = rag_service.ask_rag(request.question)
    return {"answer": answer}

@app.get("/")
async def root():
    return {"message": "博物馆文物知识问答，请使用 POST /ask 接口"}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

