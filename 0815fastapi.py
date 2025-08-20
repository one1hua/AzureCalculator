import os
import traceback
from typing import Dict, Any, Optional, Literal, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langgraph.graph import StateGraph
from langchain_core.chat_history import InMemoryChatMessageHistory

# =========================================================
# FASTAPI
# =========================================================
app = FastAPI(title="RAG (MD + CSV) with KB filter")

# =========================================================
# Azure OpenAI 설정 (본인 값으로 교체)
#   - 엔드포인트는 '베이스 URL'만! (뒤에 /openai/deployments/... 금지)
#   - 배포 이름은 OpenAI Studio/포털의 'Deployment name'
#   - 버전은 curl로 확인된 2023-05-15 로 우선 통일
# =========================================================
AZURE_API_KEY = "8D0TqUJQKj7SfHCMcRSAs5cJBSbvvQIcsuZ6QvHFKWsVsP9JyhooJQQJ99BGACNns7RXJ3w3AAABACOG0jZk"
AZURE_ENDPOINT_BASE = "https://azure-openai-price01.openai.azure.com"  # 예시
OPENAI_API_VERSION = "2023-05-15"

AZURE_CHAT_DEPLOYMENT = "gpt-4o"                 # 예시 배포명
AZURE_EMBED_DEPLOYMENT = "text-embedding-3-small"  # 예시 배포명

# =========================================================
# 모델 초기화
# =========================================================
azure_llm = AzureChatOpenAI(
    api_key=AZURE_API_KEY,
    azure_endpoint=AZURE_ENDPOINT_BASE,
    azure_deployment=AZURE_CHAT_DEPLOYMENT,
    openai_api_version=OPENAI_API_VERSION,
    temperature=0.2,
    max_tokens=800,
    request_timeout=30,
)

embedding_model = AzureOpenAIEmbeddings(
    api_key=AZURE_API_KEY,
    azure_endpoint=AZURE_ENDPOINT_BASE,
    azure_deployment=AZURE_EMBED_DEPLOYMENT,
    openai_api_version=OPENAI_API_VERSION,
)

# =========================================================
# Chroma 경로 (절대 경로 권장)
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIR = os.path.join(BASE_DIR, "chromaDB")

def make_retriever(kb: Optional[str], k: int = 3):
    """KB 메타데이터 필터 기반 리트리버 생성"""
    search_kwargs = {"k": k}
    if kb:
        search_kwargs["filter"] = {"kb": kb}
    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embedding_model
    ).as_retriever(search_kwargs=search_kwargs)

# 기본 리트리버 (필요 시 요청마다 make_retriever로 새로 만들 수 있음)
retriever_md_base = make_retriever("kb_md", k=3)
retriever_csv_base = make_retriever("kb_csv", k=3)
retriever_all_base = make_retriever(None, k=3)  # 전체(필터 없음)

# =========================================================
# 세션별 채팅 히스토리
# =========================================================
chats_by_session_id: Dict[str, InMemoryChatMessageHistory] = {}

def get_chat_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in chats_by_session_id:
        chats_by_session_id[session_id] = InMemoryChatMessageHistory()
    return chats_by_session_id[session_id]

# =========================================================
# 요청/응답 모델
# =========================================================
class QARequest(BaseModel):
    question: str
    session_id: str
    target_kb: Optional[Literal["md", "csv", "all"]] = "all"
    k: int = 3

# =========================================================
# 유틸: 문맥/참조 변환
# =========================================================
def docs_to_context(docs) -> str:
    return "\n\n---\n\n".join(d.page_content for d in docs)

def docs_to_refs(docs) -> List[Dict[str, Any]]:
    refs = []
    for d in docs:
        meta = d.metadata or {}
        refs.append({
            "kb": meta.get("kb"),
            "source": meta.get("source"),
            "row_index": meta.get("row_index"),
            "chunk_index": meta.get("chunk_index"),
            "snippet": (d.page_content[:200] + "…") if len(d.page_content) > 200 else d.page_content,
        })
    return refs

def build_prompt(context: str, question: str, history_text: str = "") -> str:
    pre = f"다음은 문맥입니다:\n{context}\n\n"
    if history_text:
        pre = f"다음은 과거 대화와 문맥입니다:\n{history_text}\n\n{pre}"
    rules = (
        "규칙:\n"
        "- 반드시 문맥에서 근거를 찾아 한국어로 간결하게 답하세요.\n"
        "- 문맥에 없으면 모른다고 답하고, 다음 탐색 방향을 제안하세요.\n"
        "- 가능하면 출처 파일/행/섹션 등의 단서를 함께 써주세요.\n\n"
    )
    return f"{pre}{rules}질문: {question}\n\n답변:"

def select_docs(question: str, target_kb: str, k: int):
    """KB 선택에 따라 문서 검색"""
    if target_kb == "md":
        return make_retriever("kb_md", k=k).invoke(question)
    if target_kb == "csv":
        return make_retriever("kb_csv", k=k).invoke(question)
    # all: 두 KB를 각각 조회 후 합치기 (간단 병합)
    docs_md = make_retriever("kb_md", k=k).invoke(question)
    docs_csv = make_retriever("kb_csv", k=k).invoke(question)
    return docs_md + docs_csv

# =========================================================
# RAG 노드 (LangGraph)
# =========================================================
def rag_node(state: Dict[str, Any]):
    mcp = state["mcp"]
    payload = mcp["payload"]

    question: str = payload["question"]
    meta: Dict[str, Any] = payload.get("metadata", {})

    session_id: str = meta.get("session_id", "default")
    target_kb: str = meta.get("target_kb", "all")
    top_k: int = int(meta.get("k", 3))

    chat_history = get_chat_history(session_id)

    print(f"🔍 질문: {question} | target_kb={target_kb} | k={top_k}")
    docs = select_docs(question, target_kb, top_k)
    print(f"📄 검색된 문서 수: {len(docs)}")

    # 빈 검색은 200으로 안내 반환
    if not docs:
        result = "문맥에서 관련 내용을 찾지 못했습니다. target_kb를 md/csv로 바꾸거나 질문을 더 구체화해 보세요."
        chat_history.add_user_message(question)
        chat_history.add_ai_message(result)
        return {
            "mcp": {
                "source": "rag_agent",
                "destination": mcp["source"],
                "intent": "answer",
                "payload": {
                    "answer": result,
                    "references": [],
                    "metadata": {"session_id": session_id, "used_kb": target_kb, "top_k": top_k}
                }
            }
        }

    history_text = "\n".join([m.content for m in chat_history.messages])
    context = docs_to_context(docs)
    prompt = build_prompt(context=context, question=question, history_text=history_text)

    try:
        result = azure_llm.invoke(prompt).content
    except Exception as e:
        print("[LLM ERROR]", e.__class__.__name__, str(e))
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"LLM error: {e.__class__.__name__}: {e}")

    chat_history.add_user_message(question)
    chat_history.add_ai_message(result)

    refs = docs_to_refs(docs)
    return {
        "mcp": {
            "source": "rag_agent",
            "destination": mcp["source"],
            "intent": "answer",
            "payload": {
                "answer": result,
                "references": refs,
                "metadata": {"session_id": session_id, "used_kb": target_kb, "top_k": top_k}
            }
        }
    }

# LangGraph 컴파일
rag_app = StateGraph(dict)
rag_app.set_entry_point("rag_node")
rag_app.add_node("rag_node", rag_node)
rag_app.set_finish_point("rag_node")
rag_app = rag_app.compile()

def handle_mcp(mcp: Dict[str, Any]) -> Dict[str, Any]:
    if mcp["destination"] == "rag_agent":
        return rag_app.invoke({"mcp": mcp})
    raise ValueError(f"Unknown destination: {mcp['destination']}")

# =========================================================
# API 스키마 & 엔드포인트
# =========================================================
class QARequest(BaseModel):
    question: str
    session_id: str
    target_kb: Optional[Literal["md", "csv", "all"]] = "all"
    k: int = 3

@app.get("/")
def health():
    return {"message": "RAG (MD + CSV) FastAPI 서버 동작 중"}

@app.post("/answer")
def answer_question(req: QARequest):
    print(f"✅ /answer 호출: target_kb={req.target_kb}, k={req.k}")
    try:
        mcp = {
            "source": "user",
            "destination": "rag_agent",
            "intent": "get_answer",
            "payload": {
                "question": req.question,
                "metadata": {
                    "session_id": req.session_id,
                    "target_kb": req.target_kb or "all",
                    "k": req.k
                }
            }
        }
        response = handle_mcp(mcp)
        return response["mcp"]["payload"]
    except HTTPException:
        raise
    except Exception as e:
