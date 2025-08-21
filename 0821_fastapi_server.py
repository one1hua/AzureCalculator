# fastapi_server.py
import os
import traceback
from typing import Dict, Any, Optional, Literal, List, Tuple
from typing_extensions import TypedDict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain.schema import Document

# ==============================
# FastAPI 앱
# ==============================
app = FastAPI(title="RAG + LLM Hybrid (MD + CSV)")

# ==============================
# Azure OpenAI 설정
# ==============================
AZURE_API_KEY       = os.environ.get("AZURE_OPENAI_API_KEY", "8D0TqUJQKj7SfHCMcRSAs5cJBSbvvQIcsuZ6QvHFKWsVsP9JyhooJQQJ99BGACNns7RXJ3w3AAABACOG0jZk")
AZURE_ENDPOINT_BASE = os.environ.get("AZURE_OPENAI_ENDPOINT", "https://azure-openai-price01.openai.azure.com").rstrip("/")
OPENAI_API_VERSION  = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

AZURE_CHAT_DEPLOY   = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o")
AZURE_EMBED_DEPLOY  = os.environ.get("AZURE_OPENAI_EMBED_DEPLOYMENT", "text-embedding-3-small")

# LLM / Embedding
azure_llm = AzureChatOpenAI(
    api_key=AZURE_API_KEY,
    azure_endpoint=AZURE_ENDPOINT_BASE,
    azure_deployment=AZURE_CHAT_DEPLOY,
    openai_api_version=OPENAI_API_VERSION,
    temperature=0.2,
    max_tokens=900,
    request_timeout=60,
)

embedding_model = AzureOpenAIEmbeddings(
    api_key=AZURE_API_KEY,
    azure_endpoint=AZURE_ENDPOINT_BASE,
    azure_deployment=AZURE_EMBED_DEPLOY,
    openai_api_version=OPENAI_API_VERSION,
)

# ==============================
# Chroma (컬렉션/경로 고정)
# ==============================
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIR = os.path.join(BASE_DIR, "chromaDB")

VECTORSTORE = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embedding_model,
    collection_name="langchain",  # 현재 데이터가 있는 컬렉션
)

# ==============================
# 세션 히스토리
# ==============================
chats_by_session_id: Dict[str, InMemoryChatMessageHistory] = {}
def get_chat_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in chats_by_session_id:
        chats_by_session_id[session_id] = InMemoryChatMessageHistory()
    return chats_by_session_id[session_id]

# ==============================
# 요청/응답 모델
# ==============================
class QARequest(BaseModel):
    question: str
    session_id: str
    target_kb: Optional[Literal["auto", "md", "csv", "all"]] = "auto"
    k: int = 4
    mode: Literal["rag_strict", "rag_assist", "llm_only"] = "rag_assist"  # ⭐ 하이브리드 모드

class QAResponse(BaseModel):
    answer: str
    references: List[Dict[str, Any]]
    metadata: Dict[str, Any]

# ==============================
# 유틸
# ==============================
def docs_to_refs(docs: List[Document]) -> List[Dict[str, Any]]:
    out = []
    for d in docs:
        m = d.metadata or {}
        out.append({
            "kb": m.get("kb"),
            "source": m.get("source"),
            "row_index": m.get("row_index"),
            "chunk_index": m.get("chunk_index"),
            "snippet": (d.page_content[:200] + "…") if len(d.page_content) > 200 else d.page_content
        })
    return out

def join_docs(docs: List[Document]) -> str:
    return "\n\n---\n\n".join(d.page_content for d in docs)

def build_prompt_rag(context: str, question: str, history_text: str = "") -> str:
    pre = f"다음은 문맥입니다:\n{context}\n\n"
    if history_text:
        pre = f"다음은 과거 대화와 문맥입니다:\n{history_text}\n\n{pre}"
    rules = (
        "규칙:\n"
        "- 우선 문맥에서 근거를 찾아 한국어로 간결하게 답하세요.\n"
        "- 문맥이 부족하면 일반 지식으로 보완하되, Azure 'Virtual Machines'와 'Azure VMware Solution'을 혼동하지 마세요.\n"
        "- 'VM'이란 표현이 나오면 기본적으로 'Azure Virtual Machines'를 의미한다고 가정하세요.\n"
        "- 가능한 경우 출처(파일/행/섹션)를 함께 언급하세요.\n"
        "- 확실하지 않은 수치나 SKU는 '대략', '예시'로 표시하세요.\n\n"
    )
    return f"{pre}{rules}질문: {question}\n\n답변:"

def build_prompt_llm_only(question: str, history_text: str = "") -> str:
    pre = ""
    if history_text:
        pre = f"다음은 과거 대화입니다:\n{history_text}\n\n"
    rules = (
        "규칙:\n"
        "- 일반 지식과 최신 추론으로 답하세요. 필요하면 예시도 제시하세요.\n"
        "- 'VM'은 'Azure Virtual Machines'로 간주하고, 'Azure VMware Solution'과 구분하세요.\n"
        "- 문서 출처가 없으므로, 사실관계를 자신있게 단정하기보다는 분명하고 실용적인 가이드를 제시하세요.\n\n"
    )
    return f"{pre}{rules}질문: {question}\n\n답변:"

# 간단 라우팅 (키워드 기반)
def simple_route(question: str) -> Literal["md", "csv", "all"]:
    q = question.lower()
    md_keys = ["애저딱칼센", "사용법", "사용 방법", "기능", "튜토리얼", "가이드"]
    if any(key in question for key in md_keys):
        return "md"
    csv_keys = ["가격", "요금", "서비스", "목록", "종류", "설명", "엔트라", "스토리지", "컴퓨트", "데이터베이스", "vm", "virtual machine"]
    if any(key in question for key in csv_keys) or any(key in q for key in ["price","service","list","category","vm","virtual machine"]):
        return "csv"
    return "all"

# 검색
def search_with_scores(kb: Optional[str], query: str, k: int) -> List[Tuple[Document, float]]:
    """
    kb: None|"md"|"csv"|"all"
    - "md"  → filter={"kb": "kb_md"}
    - "csv" → filter={"kb": "kb_csv"}
    - "all" 또는 None → 필터 없음(전체)
    """
    if kb == "md":
        filt = {"kb": "kb_md"}
    elif kb == "csv":
        filt = {"kb": "kb_csv"}
    else:
        filt = None
    print(f"[SEARCH] kb={kb}  filter={filt}")
    return VECTORSTORE.similarity_search_with_score(query, k=k, filter=filt)

# ==============================
# 헬스체크
# ==============================
@app.get("/")
def health():
    return {"message": "Hybrid RAG/LLM OK", "persist": PERSIST_DIR, "collection": "langchain"}

# ==============================
# 메인 엔드포인트
# ==============================
@app.post("/answer", response_model=QAResponse)
def answer_question(req: QARequest):
    print(f"✅ /answer: q='{req.question}', target_kb={req.target_kb}, mode={req.mode}, k={req.k}")

    # 0) KB 선택
    if req.target_kb and req.target_kb != "auto":
        kb = req.target_kb
        routed_by = "user"
    else:
        kb = simple_route(req.question)
        routed_by = "router"

    # 1) 모드별 실행
    chat_history = get_chat_history(req.session_id)
    history_text = "\n".join([m.content for m in chat_history.messages])

    # LLM only
    if req.mode == "llm_only":
        prompt = build_prompt_llm_only(req.question, history_text)
        try:
            result = azure_llm.invoke(prompt).content
            chat_history.add_user_message(req.question)
            chat_history.add_ai_message(result)
            return QAResponse(
                answer=result,
                references=[],
                metadata={"session_id": req.session_id, "mode": req.mode, "final_kb_used": None,
                          "routed_by": routed_by, "retrieved_docs_count": 0}
            )
        except Exception as e:
            print("[LLM ERROR]", e); print(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    # RAG (strict/assist 공통 1차: KB 우선 검색)
    try:
        if kb == "all":
            md_pairs  = search_with_scores("md",  req.question, req.k)
            csv_pairs = search_with_scores("csv", req.question, req.k)
            pairs = md_pairs + csv_pairs
        else:
            pairs = search_with_scores(kb, req.question, req.k)
    except Exception as e:
        print("[SEARCH ERROR]", e); print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Search error: {e}")

    docs = [doc for doc, _ in pairs]
    print(f"[SEARCH] retrieved_docs={len(docs)} (kb={kb})")

    # RAG strict: 문서 없으면 모른다고 답
    if req.mode == "rag_strict":
        if not docs:
            return QAResponse(
                answer="문맥에서 관련 내용을 찾지 못했습니다. 질문을 더 구체화하거나 다른 KB를 지정해 보세요.",
                references=[],
                metadata={"session_id": req.session_id, "mode": req.mode,
                          "final_kb_used": kb, "routed_by": routed_by, "retrieved_docs_count": 0}
            )
        # 문서 있으면 문맥 기반으로 생성
        context = join_docs(docs)
        prompt = build_prompt_rag(context, req.question, history_text)
        try:
            result = azure_llm.invoke(prompt).content
            chat_history.add_user_message(req.question)
            chat_history.add_ai_message(result)
            return QAResponse(
                answer=result,
                references=docs_to_refs(docs),
                metadata={"session_id": req.session_id, "mode": req.mode,
                          "final_kb_used": kb, "routed_by": routed_by, "retrieved_docs_count": len(docs)}
            )
        except Exception as e:
            print("[LLM ERROR]", e); print(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    # RAG assist: 문서가 적으면 자동 보강
    # 기준: 문서 수 < 2 이거나, 사용자 질문이 VM 범주인데 csv 문서가 안 잡히는 등
    need_broaden = (len(docs) < 2)
    if need_broaden and kb != "all":
        print("[ASSIST] few docs → broaden search to ALL")
        md_pairs  = search_with_scores("md",  req.question, max(2, req.k//2))
        csv_pairs = search_with_scores("csv", req.question, req.k)
        pairs = md_pairs + csv_pairs
        docs = [doc for doc, _ in pairs]
        print(f"[ASSIST] broadened_docs={len(docs)}")

    # 최종 생성: 문맥 우선 + 일반지식 보강 허용 프롬프트
    if docs:
        context = join_docs(docs)
        prompt = build_prompt_rag(context, req.question, history_text)
    else:
        # 문맥 0이면 LLM-only 프롬프트로
        prompt = build_prompt_llm_only(req.question, history_text)

    try:
        result = azure_llm.invoke(prompt).content
        chat_history.add_user_message(req.question)
        chat_history.add_ai_message(result)
        return QAResponse(
            answer=result,
            references=docs_to_refs(docs),
            metadata={"session_id": req.session_id, "mode": req.mode,
                      "final_kb_used": kb, "routed_by": routed_by, "retrieved_docs_count": len(docs)}
        )
    except Exception as e:
        print("[LLM ERROR]", e); print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")
