 fastapi_server.py
# RAG(MD+CSV) + 하이브리드 답변(rag_strict/rag_assist/llm_only)
# - 풍성한 구조화 출력(detail/예시/표)
# - KB 라우팅(auto/md/csv/all)
# - 2024-12-01-preview API 버전 사용
# - Chroma(collection_name="langchain") 고정

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
app = FastAPI(title="Hybrid RAG + LLM (MD + CSV)")

# ==============================
# Azure OpenAI 설정 (환경변수 우선)
# ==============================
AZURE_API_KEY       = os.environ.get("AZURE_OPENAI_API_KEY", "8D0TqUJQKj7SfHCMcRSAs5cJBSbvvQIcsuZ6QvHFKWsVsP9JyhooJQQJ99BGACNns7RXJ3w3AAABACOG0jZk")
AZURE_ENDPOINT_BASE = os.environ.get("AZURE_OPENAI_ENDPOINT", "https://azure-openai-price01.openai.azure.com").rstrip("/")
OPENAI_API_VERSION  = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

AZURE_CHAT_DEPLOY   = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o")                 # 챗 배포명
AZURE_EMBED_DEPLOY  = os.environ.get("AZURE_OPENAI_EMBED_DEPLOYMENT", "text-embedding-3-small")  # 임베딩 배포명

# ==============================
# LLM / Embedding
# ==============================
azure_llm = AzureChatOpenAI(
    api_key=AZURE_API_KEY,
    azure_endpoint=AZURE_ENDPOINT_BASE,         # 반드시 베이스 URL
    azure_deployment=AZURE_CHAT_DEPLOY,
    openai_api_version=OPENAI_API_VERSION,
    temperature=0.2,
    # 기본 상한: 요청마다 더 줄 수도 있음
    max_tokens=1500,
    request_timeout=60,
)

embedding_model = AzureOpenAIEmbeddings(
    api_key=AZURE_API_KEY,
    azure_endpoint=AZURE_ENDPOINT_BASE,         # 반드시 베이스 URL
    azure_deployment=AZURE_EMBED_DEPLOY,
    openai_api_version=OPENAI_API_VERSION,
)

# ==============================
# Chroma (컬렉션/경로 고정)
# ==============================
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIR  = os.path.join(BASE_DIR, "chromaDB")
COLLECTION   = "langchain"  # 실제 데이터가 있는 컬렉션명

VECTORSTORE = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embedding_model,
    collection_name=COLLECTION,
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
    k: int = 6
    mode: Literal["rag_strict", "rag_assist", "llm_only"] = "rag_assist"

    # 풍성도 옵션
    detail: Literal["concise", "standard", "rich"] = "rich"
    max_tokens: int = 1500
    include_examples: bool = True
    include_table: bool = True

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

def _structure_instructions(detail: str, include_examples: bool, include_table: bool) -> str:
    # 분량 가이드
    if detail == "concise":
        length_rule = "- 길이는 4~6문장 정도로 간결하게.\n"
    elif detail == "standard":
        length_rule = "- 길이는 8~12문장, 필요한 소제목 포함.\n"
    else:  # rich
        length_rule = "- 길이는 최소 12~20문장, 소제목/목록을 적극 활용.\n"

    # 필요한 섹션만 남김
    parts = [
        "1) 서비스 요약 (2~3문장)",
        "2) 핵심 개념/정의",
        "3) 주요 기능/옵션 (불릿 목록)",
        "4) 다음 단계/추가 자료",
    ]

    return (
        "표현 규칙:\n"
        f"{length_rule}"
        "- 반드시 한국어로 명확하고 단정적으로 작성.\n"
        "- 문맥(근거)와 일반 지식을 결합하되, Azure Virtual Machines와 Azure VMware Solution을 혼동하지 말 것.\n"
        "- 수치/가격은 문맥에 없으면 '예시/대략'으로 표기.\n"
        + "\n".join(f"- {p}" for p in parts) + "\n"
    )



def build_prompt_rag(context: str, question: str, history_text: str, req: QARequest) -> str:
    header = "다음은 참조할 문맥입니다:\n" + context + "\n\n"
    history = f"이전 대화:\n{history_text}\n\n" if history_text else ""
    structure = _structure_instructions(req.detail, req.include_examples, req.include_table)
    guardrails = (
        "추가 규칙:\n"
        "- 문맥이 충분하면 문맥을 최우선 근거로 사용.\n"
        "- 문맥이 불충분한 부분은 일반 지식으로 보완.\n"
        "- 'VM'은 기본적으로 Azure Virtual Machines를 의미.\n"
        "- 가능 시 근거의 출처(파일/행/섹션)를 자연스럽게 언급.\n"
    )
    return f"{history}{header}{structure}{guardrails}\n사용자 질문: {question}\n\n구조화된 답변을 작성하세요."

def build_prompt_llm_only(question: str, history_text: str, req: QARequest) -> str:
    history = f"이전 대화:\n{history_text}\n\n" if history_text else ""
    structure = _structure_instructions(req.detail, req.include_examples, req.include_table)
    guardrails = (
        "추가 규칙:\n"
        "- 일반 지식과 실무 경험칙을 활용해 실용적으로 설명.\n"
        "- 'VM'은 Azure Virtual Machines로 간주, Azure VMware Solution과 구분.\n"
    )
    return f"{history}{structure}{guardrails}\n사용자 질문: {question}\n\n구조화된 답변을 작성하세요."

def call_llm_with_fallback(prompt: str, max_tokens: int) -> str:
    """일부 환경에서 invoke에 max_tokens 전달이 안 될 수 있어 안전 처리."""
    try:
        return azure_llm.invoke(prompt, max_tokens=max_tokens).content
    except TypeError:
        # langchain 버전에 따라 호출 시그니처가 다를 수 있음 → 기본 max_tokens로 호출
        return azure_llm.invoke(prompt).content

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
    return {
        "message": "Hybrid RAG/LLM OK",
        "persist": PERSIST_DIR,
        "collection": COLLECTION,
        "api_version": OPENAI_API_VERSION,
    }

# ==============================
# 메인 엔드포인트
# ==============================
@app.post("/answer", response_model=QAResponse)
def answer_question(req: QARequest):
    print(f"✅ /answer: q='{req.question}', target_kb={req.target_kb}, mode={req.mode}, k={req.k}, detail={req.detail}")

    # 0) KB 선택
    if req.target_kb and req.target_kb != "auto":
        kb = req.target_kb
        routed_by = "user"
    else:
        kb = simple_route(req.question)
        routed_by = "router"

    chat_history = get_chat_history(req.session_id)
    history_text = "\n".join([m.content for m in chat_history.messages])

    # 1) LLM-only 모드
    if req.mode == "llm_only":
        prompt = build_prompt_llm_only(req.question, history_text, req)
        try:
            result = call_llm_with_fallback(prompt, req.max_tokens)
            chat_history.add_user_message(req.question)
            chat_history.add_ai_message(result)
            return QAResponse(
                answer=result,
                references=[],
                metadata={
                    "session_id": req.session_id,
                    "mode": req.mode,
                    "final_kb_used": None,
                    "routed_by": routed_by,
                    "retrieved_docs_count": 0,
                }
            )
        except Exception as e:
            print("[LLM ERROR]", e); print(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    # 2) RAG 검색 (strict/assist 공통 1차)
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

    # 3) RAG strict: 문서 없으면 모른다고 답
    if req.mode == "rag_strict":
        if not docs:
            return QAResponse(
                answer="문맥에서 관련 내용을 찾지 못했습니다. 질문을 더 구체화하거나 다른 KB를 지정해 보세요.",
                references=[],
                metadata={
                    "session_id": req.session_id,
                    "mode": req.mode,
                    "final_kb_used": kb,
                    "routed_by": routed_by,
                    "retrieved_docs_count": 0,
                }
            )
        context = join_docs(docs)
        prompt = build_prompt_rag(context, req.question, history_text, req)
        try:
            result = call_llm_with_fallback(prompt, req.max_tokens)
            chat_history.add_user_message(req.question)
            chat_history.add_ai_message(result)
            return QAResponse(
                answer=result,
                references=docs_to_refs(docs),
                metadata={
                    "session_id": req.session_id,
                    "mode": req.mode,
                    "final_kb_used": kb,
                    "routed_by": routed_by,
                    "retrieved_docs_count": len(docs),
                }
            )
        except Exception as e:
            print("[LLM ERROR]", e); print(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    # 4) RAG assist: 문서가 적으면 자동 보강(ALL로 확장)
    need_broaden = (len(docs) < 2)
    if need_broaden and kb != "all":
        print("[ASSIST] few docs → broaden search to ALL")
        md_pairs  = search_with_scores("md",  req.question, max(2, req.k // 2))
        csv_pairs = search_with_scores("csv", req.question, req.k)
        pairs = md_pairs + csv_pairs
        docs = [doc for doc, _ in pairs]
        print(f"[ASSIST] broadened_docs={len(docs)}")

    # 5) 최종 생성: 문맥 우선 + 일반지식 보강 허용
    if docs:
        context = join_docs(docs)
        prompt = build_prompt_rag(context, req.question, history_text, req)
    else:
        prompt = build_prompt_llm_only(req.question, history_text, req)

    try:
        result = call_llm_with_fallback(prompt, req.max_tokens)
        chat_history.add_user_message(req.question)
        chat_history.add_ai_message(result)
        return QAResponse(
            answer=result,
            references=docs_to_refs(docs),
            metadata={
                "session_id": req.session_id,
                "mode": req.mode,
                "final_kb_used": kb,
                "routed_by": routed_by,
                "retrieved_docs_count": len(docs),
            }
        )
    except Exception as e:
        print("[LLM ERROR]", e); print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

