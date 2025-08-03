from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langgraph.graph import StateGraph
from langchain_core.chat_history import InMemoryChatMessageHistory

# FASTAPI 설정
app = FastAPI()

# API Version 설정 (최상단에 위치)
API_VERSION = "2024-12-01-preview"

# Azure OpenAI 설정
AZURE_CHAT_ENDPOINT = "https://azure-openai-price01.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2025-01-01-preview"
AZURE_CHAT_DEPLOYMENT = "gpt-4o"
AZURE_EMBEDDING_ENDPOINT = "https://azure-openai-price01.openai.azure.com/openai/deployments/text-embedding-3-small/embeddings?api-version=2023-05-15"
AZURE_EMBEDDING_DEPLOYMENT = "text-embedding-3-small"
AZURE_API_KEY = "8D0TqUJQKj7SfHCMcRSAs5cJBSbvvQIcsuZ6QvHFKWsVsP9JyhooJQQJ99BGACNns7RXJ3w3AAABACOG0jZk"


# 모델 초기화
azure_llm = AzureChatOpenAI(
    api_key=AZURE_API_KEY,
    azure_endpoint=AZURE_CHAT_ENDPOINT,
    azure_deployment=AZURE_CHAT_DEPLOYMENT,
    openai_api_version=API_VERSION,
    temperature=0.3,
    max_tokens=800
)

embedding_model = AzureOpenAIEmbeddings(
    azure_endpoint=AZURE_EMBEDDING_ENDPOINT,
    azure_deployment=AZURE_EMBEDDING_DEPLOYMENT,
    api_key=AZURE_API_KEY,
    openai_api_version=API_VERSION
)


# Azure OpenAI 설정
#AZURE_ENDPOINT = "https://azure-openai-price01.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2025-01-01-preview"
#AZURE_API_KEY = "8D0TqUJQKj7SfHCMcRSAs5cJBSbvvQIcsuZ6QvHFKWsVsP9JyhooJQQJ99BGACNns7RXJ3w3AAABACOG0jZk"
#API_VERSION = "2024-12-01-preview"
#EMBEDDING_DEPLOYMENT = "text-embedding-3-small"
#CHAT_DEPLOYMENT = "gpt-4o"

# 모델 초기화
#azure_llm = AzureChatOpenAI(
#    api_key=AZURE_API_KEY,
#    azure_endpoint=AZURE_ENDPOINT,
#    azure_deployment=CHAT_DEPLOYMENT,
#    openai_api_version=API_VERSION,
#    temperature=0.3,
#    max_tokens=800
#)

#embedding_model = AzureOpenAIEmbeddings(
#    azure_endpoint=AZURE_ENDPOINT,
#    azure_deployment=EMBEDDING_DEPLOYMENT,
#    api_key=AZURE_API_KEY,
#    openai_api_version=API_VERSION
#)

# Chroma 벡터스토어 로딩
retriever = Chroma(
    persist_directory="./chromaDB",
    embedding_function=embedding_model
).as_retriever(search_kwargs={"k": 3})

# 세션별 채팅 히스토리
chats_by_session_id = {}
def get_chat_history(session_id: str):
    if session_id not in chats_by_session_id:
        chats_by_session_id[session_id] = InMemoryChatMessageHistory()
    return chats_by_session_id[session_id]

# RAG 노드 (Retrieval + LLM 호출)
def rag_node(state: Dict[str, Any]):
    mcp = state["mcp"]
    question = mcp["payload"]["question"]
    session_id = mcp["payload"]["metadata"]["session_id"]
    chat_history = get_chat_history(session_id)

    print(f"🔍 질문: {question}")
    docs = retriever.invoke(question)
    print(f"📄 검색된 문서 수: {len(docs)}")

    if not docs:
        raise ValueError("❌ 검색된 문서가 없습니다.")

    history_text = "\n".join([m.content for m in chat_history.messages])
    context = "\n".join(doc.page_content for doc in docs)

    full_context = f"{history_text}\n\n{context}" if history_text else context
    prompt = f"""다음은 문맥입니다:\n{full_context}\n\n질문: {question}\n\n답변:"""

    result = azure_llm.invoke(prompt).content

    chat_history.add_user_message(question)
    chat_history.add_ai_message(result)

    return {
        "mcp": {
            "source": "rag_agent",
            "destination": mcp["source"],
            "intent": "answer",
            "payload": {
                "answer": result,
                "references": [doc.page_content for doc in docs],
                "metadata": {"session_id": session_id}
            }
        }
    }

# LangGraph 컴파일
rag_app = StateGraph(dict)
rag_app.set_entry_point("rag_node")
rag_app.add_node("rag_node", rag_node)
rag_app.set_finish_point("rag_node")
rag_app = rag_app.compile()

# MCP 핸들러
def handle_mcp(mcp: Dict[str, Any]) -> Dict[str, Any]:
    if mcp["destination"] == "rag_agent":
        return rag_app.invoke({"mcp": mcp})
    else:
        raise ValueError(f"Unknown destination: {mcp['destination']}")

# Request Body
class QARequest(BaseModel):
    question: str
    session_id: str

# 헬스체크
@app.get("/")
def health():
    return {"message": "Azure RAG FastAPI 서버 동작 중"}

# 질문 엔드포인트
@app.post("/answer")
def answer_question(req: QARequest):
    print(f"✅ /answer API 호출됨: {req}")
    try:
        intent = "get_answer"
        mcp = {
            "source": "user",
            "destination": "rag_agent",
            "intent": intent,
            "payload": {
                "question": req.question,
                "metadata": {
                    "session_id": req.session_id
                }
            }
        }
        response = handle_mcp(mcp)
        return response["mcp"]["payload"]
    except Exception as e:
        print(f"❌ Error while processing /answer: {e}")
        raise HTTPException(status_code=500, detail=str(e))

