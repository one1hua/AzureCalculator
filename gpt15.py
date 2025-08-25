from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, TypedDict
from langchain.schema import Document, SystemMessage, HumanMessage
from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langgraph.graph import StateGraph, END, START
from langchain_core.chat_history import InMemoryChatMessageHistory
import uuid
import pandas as pd
import os
import sqlite3
import logging
import json
import re
from contextlib import asynccontextmanager
from fastapi.responses import JSONResponse

from datetime import datetime, timezone, timedelta
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Body
from fastapi.responses import StreamingResponse
import io
import csv
import copy

# ---- logging ----
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# 파일 상단 전역
SESSION_PAYLOADS: Dict[str, Dict[str, Any]] = {}
pending_by_session_id: Dict[str, Dict[str, Any]] = {}

# ---- 기본 기간(묻지 않고 바로 계산용) ----
DEFAULT_MONTH_HOURS = 720   # 1달 = 720시간으로 고정
DEFAULT_MONTHS = 1          # 디스크 등 월 단위는 1개월로 고정

# ---- 종량제(Consumption) 장바구니 추가 시 할인율 ----
CONSUMPTION_DISCOUNT_RATE = 0.10

# ---- SQLite ----
DATABASE_NAME = "azure_price_data.db"
CSV_FILES = [
    "data/azure_vm_prices_filtered_specs_koreacentral.csv",
    "data/azure_disk_prices_filtered_specs_koreacentral.csv",
    "data/filtered_db_compute_prices.csv",
    "data/0821_databricks_prices_koreacentral.csv",
    "data/0821_aoai_prices_koreacentral.csv"
]

def create_and_load_db():
    logging.info("SQLite 데이터베이스 초기화 및 데이터 로드 시작...")
    if os.path.exists(DATABASE_NAME):
        os.remove(DATABASE_NAME)
        logging.info(f"기존 데이터베이스 '{DATABASE_NAME}' 삭제")

    conn = sqlite3.connect(DATABASE_NAME)
    try:
        for file_name in CSV_FILES:
            try:
                df = pd.read_csv(file_name, encoding='utf-8-sig', on_bad_lines='skip')
                table_name = os.path.splitext(os.path.basename(file_name))[0]
                # 컬럼명 정제
                df.columns = df.columns.str.replace('[^A-Za-z0-9_]+', '', regex=True)
                df.to_sql(table_name, conn, if_exists='replace', index=False)
                logging.info(f"'{file_name}' -> 테이블 '{table_name}' 적재")
            except FileNotFoundError:
                logging.error(f"파일 없음: {file_name}")
            except Exception as e:
                logging.exception(f"{file_name} 로드 오류: {e}")
    finally:
        conn.close()
        logging.info("SQLite 데이터베이스 로드 완료.")

# ---- lifespan ----
@asynccontextmanager
async def lifespan(app: FastAPI):
    create_and_load_db()
    yield

app = FastAPI(lifespan=lifespan)

# CORS (Streamlit/Gradio 등 프론트에서 직접 호출할 수 있게)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 필요 시 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- 환율 -----
KRW_RATE = float(os.getenv("USD_KRW", "1350"))  # .env에서 USD_KRW=1350 처럼 설정
MONTHLY_HOURS = 730  # 예약 단가가 hourly일 때 1개월 환산시 사용

def to_krw(usd: float) -> int:
    try:
        return int(round(float(usd) * KRW_RATE))
    except Exception:
        return 0

def fmt_money_krw(v_krw: float) -> str:
    return f"{int(round(v_krw)):,}원"

def parse_int_first(text: str, default: int = 0) -> int:
    m = re.search(r"(\d+)", text or "")
    return int(m.group(1)) if m else default

def parse_hours(text: str, default: int = 0) -> int:
    """
    '100시간', '200 h', '300hour' 등에서 시간 정수 추출
    """
    m = re.search(r'(\d{1,7})\s*(?:시간|hour|hours|hr|h)\b', text.lower())
    if m:
        return int(m.group(1))
    # 숫자만 있는 경우도 허용
    return parse_int_first(text, default)

# 환율 유틸 아래에 추가
def fmt_usd(v: float) -> str:
    try:
        return f"${float(v):,.4f}"
    except Exception:
        return "$0.0000"

def both_prices_line(usd: float, uom: str|None) -> str:
    """시간 단가(Consumption) 표기: USD/h 와 KRW/h 동시 표시"""
    if not usd:
        return "N/A"
    krw = to_krw(usd)
    u = (uom or "1 Hour")
    return f"{fmt_usd(usd)} / {u}  (≈ {fmt_money_krw(krw)} / {u}, 환율 1 USD = {int(KRW_RATE):,}원)"

def both_prices_total(usd: float, label: str) -> str:
    """예약제 총액 표기: USD와 KRW 동시 표시 + 전체 기간 라벨"""
    if not usd:
        return "N/A"
    krw = to_krw(usd)
    return f"{fmt_usd(usd)} (≈ {fmt_money_krw(krw)}) / 전체 {label}"

def parse_months(text: str, default: int = 0) -> int:
    m = re.search(r'(\d{1,6})\s*(?:개월|month|months|mo)\b', (text or "").lower())
    if m:
        return int(m.group(1))
    # 숫자만 온 경우도 허용
    m2 = re.search(r'(\d{1,6})', (text or "").lower())
    return int(m2.group(1)) if m2 else default

def normalize_db_keyword(s: str) -> str:
    """사용자 텍스트/LLM 파싱 결과를 Azure 상품명 키워드로 정규화"""
    s = (s or "").lower()
    if "mysql" in s:
        return "MySQL"
    if "maria" in s:
        return "MariaDB"   # (필요시 MariaDB도 테이블에 있으면 매칭)
    if "postgre" in s:
        return "PostgreSQL"
    if "sql" in s:
        # 'sql' 만 온 경우는 Azure SQL Database로 가정
        return "SQL Database"
    return s

def normalize_disk_category(s: str) -> str:
    s = (s or "").lower()
    if "hdd" in s:
        return "Standard HDD"
    # default: ssd
    return "Standard SSD"

def reset_session(session_id: str):
    SESSION_PAYLOADS.pop(session_id, None)
    pending_by_session_id.pop(session_id, None)
    shopping_carts.pop(session_id, None)

# ========= [헬퍼] 공통 컬럼 유틸 =========
def _has_cols(conn, table_name: str, cols: List[str]) -> Dict[str, bool]:
    cur = conn.cursor()
    cur.execute(f'PRAGMA table_info("{table_name}")')
    present = {r[1] for r in cur.fetchall()}
    return {c: (c in present) for c in cols}

def _first_col_name(conn, table_name: str, candidates: List[str]) -> str | None:
    cur = conn.cursor()
    cur.execute(f'PRAGMA table_info("{table_name}")')
    present = {r[1] for r in cur.fetchall()}
    for c in candidates:
        if c in present:
            return c
    return None

# ==== [NEW] case-insensitive column helpers ====
def cols_of(conn, table_name: str) -> list[str]:
    cur = conn.cursor()
    return [r[1] for r in cur.execute(f'PRAGMA table_info("{table_name}")').fetchall()]

def find_col_ci(cols: list[str], *candidates: str) -> str | None:
    low = {c.lower(): c for c in cols}
    for cand in candidates:
        k = cand.lower()
        if k in low:
            return low[k]
    # 공백/하이픈 제거 매칭도 시도
    strip = {re.sub(r'[^a-z0-9]', '', c.lower()): c for c in cols}
    for cand in candidates:
        k = re.sub(r'[^a-z0-9]', '', cand.lower())
        if k in strip:
            return strip[k]
    return None

# ==== [NEW] DB three-price picker by ServiceName ====
def fetch_db_three_prices_by_service(conn, table_name: str, service_key: str) -> dict:
    """
    filtered_db_compute_prices에서 ServiceName 기준으로
    Consumption / Reservation(1Y,3Y) 한 줄씩 뽑는다.
    - PriceType: Consumption / Reservation (다양 표기 허용)
    - ReservationTerm: '1 Year', '3 Year' 등 변형 허용
    """
    cur = conn.cursor()
    cols = cols_of(conn, table_name)

    service_col = find_col_ci(cols, "serviceName", "Service")
    meter_col   = find_col_ci(cols, "productName")
    price_col   = find_col_ci(cols, "retailPrice")
    uom_col     = find_col_ci(cols, "unitOfMeasure")
    pricetype_col = find_col_ci(cols, "type")
    term_col    = find_col_ci(cols, "reservationTerm")

    if not service_col or not price_col:
        return {"consumption": None, "reservation_1y": None, "reservation_3y": None}

    def _rowdict():
        dcols = [d[0] for d in cur.description]
        return lambda r: dict(zip(dcols, r)) if r else None

    # 공통 where: ServiceName LIKE %key%
    where_service = f'{service_col} LIKE ?'
    p_service = [f'%{service_key}%']

    # 1) Consumption
    cons = None
    if pricetype_col:
        q = f'''SELECT * FROM "{table_name}"
                WHERE {where_service}
                  AND LOWER({pricetype_col}) IN ('consumption','on-demand','ondemand')
                ORDER BY {price_col} ASC LIMIT 1'''
        cur.execute(q, p_service)
        cons = _rowdict()(cur.fetchone())
    if not cons:
        # priceType 없이도 잡히게 가장 싼 거 하나라도
        q = f'''SELECT * FROM "{table_name}"
                WHERE {where_service}
                ORDER BY {price_col} ASC LIMIT 1'''
        cur.execute(q, p_service)
        cons = _rowdict()(cur.fetchone())

    # 2) Reservation 1Y
    r1y = None
    if pricetype_col and term_col:
        q = f'''SELECT * FROM "{table_name}"
                WHERE {where_service}
                  AND LOWER({pricetype_col}) IN ('reservation','reserved','ri','reservation linux','reservation windows')
                  AND LOWER({term_col}) IN ('1 year','1y','12 months','12 month')
                ORDER BY {price_col} ASC LIMIT 1'''
        cur.execute(q, p_service)
        r1y = _rowdict()(cur.fetchone())
    if not r1y and pricetype_col:
        # PriceType에 연 수가 들어간 케이스
        q = f'''SELECT * FROM "{table_name}"
                WHERE {where_service}
                  AND LOWER({pricetype_col}) IN ('reservation 1 year','reserved 1 year','ri 1 year')
                ORDER BY {price_col} ASC LIMIT 1'''
        cur.execute(q, p_service)
        r1y = _rowdict()(cur.fetchone())

    # 3) Reservation 3Y
    r3y = None
    if pricetype_col and term_col:
        q = f'''SELECT * FROM "{table_name}"
                WHERE {where_service}
                  AND LOWER({pricetype_col}) IN ('reservation','reserved','ri','reservation linux','reservation windows')
                  AND LOWER({term_col}) IN ('3 years')
                ORDER BY {price_col} ASC LIMIT 1'''
        cur.execute(q, p_service)
        r3y = _rowdict()(cur.fetchone())
    if not r3y and pricetype_col:
        q = f'''SELECT * FROM "{table_name}"
                WHERE {where_service}
                  AND LOWER({pricetype_col}) IN ('reservation 3 years','reserved 3 year','ri 3 year','reservation 3 years','reserved 3 years','ri 3 years')
                ORDER BY {price_col} ASC LIMIT 1'''
        cur.execute(q, p_service)
        r3y = _rowdict()(cur.fetchone())

    # 표시용 sku/meter/uom 보정
    def _pack(row):
        if not row: return None
        return {
            "sku": (row.get(meter_col) or row.get(service_col) or "DB").strip(),
            "usd": float(row.get(price_col) or 0.0),
            "uom": (row.get(uom_col) or "1 Hour").strip()
        }

    return {
        "consumption": _pack(cons),
        "reservation_1y": _pack(r1y),
        "reservation_3y": _pack(r3y)
    }

# ========= [헬퍼] DB: serviceName 기준 3단가(Consumption / Reservation 1Y / 3Y) =========
def fetch_db_prices_by_service(conn, table_name: str, service_key: str) -> dict:
    """
    service_key 예:
      - SQL Database
      - Azure Database for MySQL
      - Azure Database for PostgreSQL
    스키마 가정:
      - serviceName, type(=PriceType 유사), reservationTerm(1 Year/3 Year), RetailPrice, UnitOfMeasure, SkuName/MeterName/ProductName
    """
    cur = conn.cursor()
    cols = _has_cols(conn, table_name, ["serviceName", "type", "reservationTerm", "RetailPrice", "UnitOfMeasure", "SkuName", "MeterName", "ProductName"])
    if not cols["serviceName"] or not cols["type"] or not cols["RetailPrice"]:
        return {}

    name_cols = [c for c in ["SkuName", "MeterName", "ProductName"] if cols.get(c)]
    name_expr = "COALESCE(" + ",".join([c for c in name_cols]) + ")"

    def _rowdict():
        cns = [d[0] for d in cur.description]
        return lambda r: dict(zip(cns, r)) if r else None

    # 1) Consumption 한 줄
    q1 = f'''
      SELECT {name_expr} AS sku, RetailPrice, UnitOfMeasure
      FROM "{table_name}"
      WHERE serviceName LIKE ? AND type = 'Consumption'
      ORDER BY RetailPrice ASC LIMIT 1
    '''
    cur.execute(q1, (f"%{service_key}%",))
    cons = _rowdict()(cur.fetchone())

    # 2) Reservation 1Y
    res1 = None
    if cols.get("reservationTerm"):
        q2 = f'''
          SELECT {name_expr} AS sku, RetailPrice, UnitOfMeasure, reservationTerm
          FROM "{table_name}"
          WHERE serviceName LIKE ? AND type IN ('Reservation','Reserved')
                AND reservationTerm IN ('1 Year','1 year','1Y','12 Months','12 Month')
          ORDER BY RetailPrice ASC LIMIT 1
        '''
        cur.execute(q2, (f"%{service_key}%",))
        res1 = _rowdict()(cur.fetchone())

    # 3) Reservation 3Y
    res3 = None
    if cols.get("reservationTerm"):
        q3 = f'''
          SELECT {name_expr} AS sku, RetailPrice, UnitOfMeasure, reservationTerm
          FROM "{table_name}"
          WHERE serviceName LIKE ? AND type IN ('Reservation','Reserved')
                AND reservationTerm IN ('3 Years','3 Year','3 year','3Y','36 Months','36 Month')
          ORDER BY RetailPrice ASC LIMIT 1
        '''
        cur.execute(q3, (f"%{service_key}%",))
        res3 = _rowdict()(cur.fetchone())

    return {
        "consumption": cons,
        "reservation_1y": res1,
        "reservation_3y": res3,
    }

# ========= [헬퍼] Databricks: Standard/Premium DBU 시간단가 =========
def fetch_databricks_tiers(conn, table_name: str) -> dict:
    """
    MeterName에서
      - 'Standard All-purpose Compute DBU'
      - 'Premium All-purpose Compute DBU'
    를 찾아 RetailPrice(시간 단가)와 UnitOfMeasure 반환
    (열 이름은 대소문자/특수문자 차이를 흡수)
    """
    cur = conn.cursor()
    cols = cols_of(conn, table_name)

    meter_col = find_col_ci(cols, "MeterName")
    price_col = find_col_ci(cols, "RetailPrice")
    uom_col   = find_col_ci(cols, "UnitOfMeasure")

    if not (meter_col and price_col):
        return {}

    def _get_one(keyword: str):
        q = f'''
          SELECT {meter_col} AS MeterName, {price_col} AS RetailPrice, {uom_col} AS UnitOfMeasure
          FROM "{table_name}"
          WHERE {meter_col} LIKE ?
          ORDER BY {price_col} ASC
          LIMIT 1
        '''
        cur.execute(q, (f"%{keyword}%",))
        r = cur.fetchone()
        if not r: return None
        dcols = [d[0] for d in cur.description]
        return dict(zip(dcols, r))

    std = _get_one("Standard All-purpose Compute DBU")
    prm = _get_one("Premium All-purpose Compute DBU")
    return {"standard": std, "premium": prm}


# ========= [헬퍼] AOAI: 모델별 Input/Output 단가(보통 1K tokens 단위) =========
def fetch_aoai_inout_prices(conn, table_name: str, model_key: str) -> dict:
    """
    AOAI 가격표에서 모델별 Input/Output 단가를 읽는다.
    - 컬럼명은 대소문자/표기 차이를 흡수(find_col_ci)
    - metername 내 'Inp'/'Outp' 약어도 지원
    - 반환 dict 키는 spec_matching_node에서 기대하는 표준 키로 통일:
      {"input": {"MeterName", "RetailPrice", "UnitOfMeasure"},
       "output": {...}}
    """
    cur = conn.cursor()

    # 현재 테이블의 실제 컬럼명들을 가져와 케이스-무관 매칭
    cols = cols_of(conn, table_name)
    meter_col = find_col_ci(cols, "metername", "MeterName")
    price_col = find_col_ci(cols, "retailprice", "RetailPrice")
    uom_col   = find_col_ci(cols, "unitofmeasure", "UnitOfMeasure")
    if not meter_col or not price_col:
        return {}

    # 모델 키워드(예: gpt-4o) 완화: 공백/언더바/대시 섞여도 LIKE 매칭되게
    model_like = f"%{model_key.replace('_',' ').replace('-', ' ')}%"
    # 실제 metername에는 하이픈이 많으니 원형도 같이 시도
    model_like_b = f"%{model_key}%"

    def _one(kind_keys: list[str]):
        # kind_keys 예: ["Input","Inp"] / ["Output","Outp"]
        ors = " OR ".join([f"{meter_col} LIKE ?" for _ in kind_keys])
        q = f'''
          SELECT
            {meter_col} AS MeterName,
            {price_col} AS RetailPrice,
            {uom_col}   AS UnitOfMeasure
          FROM "{table_name}"
          WHERE ({meter_col} LIKE ? OR {meter_col} LIKE ?)
            AND ({ors})
          ORDER BY {price_col} ASC
          LIMIT 1
        '''
        params = [model_like, model_like_b] + [f"%{k}%" for k in kind_keys]
        cur.execute(q, params)
        r = cur.fetchone()
        if not r:
            return None
        dcols = [d[0] for d in cur.description]  # ['MeterName','RetailPrice','UnitOfMeasure']
        row = dict(zip(dcols, r))
        # 기본 UoM 보정
        if not row.get("UnitOfMeasure"):
            row["UnitOfMeasure"] = "1K tokens"
        return row

    inp = _one(["Input", "Inp", "IB"])
    outp = _one(["Output", "Outp", "OB"])
    return {"input": inp, "output": outp}

SERVICE_GUIDE = {
    "vm": (
        "서버(가상머신) 견적을 위해 아래 정보를 알려주세요:\n"
        "- CPU/코어 수 (예: 4코어)\n"
        "- RAM 용량 (예: 16GB)\n"
        "- 운영체제(OS) (예: Linux/Windows)\n"
    ),
    "db": (
        "Database 견적을 위해 아래 정보를 알려주세요:\n"
        "- Database 유형 (Azure SQL / Azure MySQL / Azure PostgreSQL)\n"
        "- 사용 시간(시간 단위, 예: 3600시간)\n"
    ),
    "disk": (
        "스토리지 견적을 위해 아래 정보를 알려주세요:\n"
        "- 디스크 종류 (SSD/HDD)\n"
        "- 용량 (예: 500GB)\n"
    ),
    "databricks": (
        "Databricks 견적을 위해 아래 정보를 알려주세요:\n"
        "- 유형 (Standard / Premium)\n"
        "- 사용 시간(예: 3600시간)"
    ),
    "aoai": (
        "Azure OpenAI(AOAI) 견적을 위해 아래 정보를 알려주세요:\n"
        "- 모델 (gpt-4o / gpt-4o-mini 등)\n"
        "- 토큰 사용량 (입력/출력: 예 1,000,000 IB / 500,000 OB)"
    ),
}

# VM만 요금제(Consumption/Reservation) 선택 플로우 사용
SERVICE_NEEDS_PLAN = {
    "vm": True,
    "disk": False,
    "db": False,
    "databricks": False,
    "aoai": False,
}

# ---- Azure OpenAI ----
# 반드시 루트 엔드포인트 사용
AZURE_ENDPOINT = "https://azure-openai-price01.openai.azure.com"
AZURE_API_KEY = "8D0TqUJQKj7SfHCMcRSAs5cJBSbvvQIcsuZ6QvHFKWsVsP9JyhooJQQJ99BGACNns7RXJ3w3AAABACOG0jZk"
API_VERSION = "2025-01-01-preview"
CHAT_DEPLOYMENT = "gpt-4o"

EMBEDDING_ENDPOINT = "https://azure-openai-price01.openai.azure.com"
EMBEDDING_API_KEY = "8D0TqUJQKj7SfHCMcRSAs5cJBSbvvQIcsuZ6QvHFKWsVsP9JyhooJQQJ99BGACNns7RXJ3w3AAABACOG0jZk"
EMBEDDING_DEPLOYMENT = "text-embedding-3-small"
EMBEDDING_API_VERSION = "2023-05-15"

azure_llm = AzureChatOpenAI(
    api_key=AZURE_API_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    azure_deployment=CHAT_DEPLOYMENT,
    openai_api_version=API_VERSION,
    temperature=0.3,
    max_tokens=800,
)

embedding_model = AzureOpenAIEmbeddings(
    azure_endpoint=EMBEDDING_ENDPOINT,
    azure_deployment=EMBEDDING_DEPLOYMENT,
    api_key=EMBEDDING_API_KEY,
    openai_api_version=EMBEDDING_API_VERSION,
)

# ---- Chroma retriever ----
retriever = Chroma(
    persist_directory="./chromaDB",
    embedding_function=embedding_model
).as_retriever(search_kwargs={"k": 5})

# ---- 세션 저장소 ----
chats_by_session_id: Dict[str, InMemoryChatMessageHistory] = {}
shopping_carts: Dict[str, List[Dict[str, Any]]] = {}
cart_meta: Dict[str, Dict[str, Any]] = {}   # 예: {"months": 3}

def get_chat_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in chats_by_session_id:
        chats_by_session_id[session_id] = InMemoryChatMessageHistory()
    return chats_by_session_id[session_id]

# ---- LangGraph 상태 ----
class GraphState(TypedDict):
    mcp: Dict[str, Any]
    last_agent: str

# --- yes/no helpers ---
def user_says_yes(text: str) -> bool:
    t = (text or "").strip().lower()
    yes_words = ["네", "예", "넵", "그래", "좋아", "추가", "담아", "넣어", "yes", "y"]
    no_words  = ["아니", "아니오", "아뇨", "노", "no", "n", "안해", "취소"]
    # 부정이 포함되면 우선 부정
    if any(w in t for w in no_words):
        return False
    return any(w in t for w in yes_words)

def user_says_no(text: str) -> bool:
    t = (text or "").strip().lower()
    return any(w in t for w in ["아니", "아니오", "아뇨", "노", "no", "n", "안해", "취소"])

# ----- [추가] 동일 SKU의 3개 요금(Consumption / Reservation 1 Year / Reservation 3 Year) 조회 -----
def fetch_three_prices_for_sku(conn, table_name: str, picked_sku: str) -> dict:
    """
    동일 SKU의 Consumption / Reservation 1Y / Reservation 3Y를 '유연 매칭'으로 조회.
    - 우선순위: SkuName 정확일치 → SkuName LIKE → ArmSkuName/ArmSku LIKE → ProductName 근사
    - 1Y/3Y 인식: PriceType 직접 표기, 또는 PriceType='Reservation' + Term 계열 컬럼('Term','ReservationTerm','TermLength' 등)로 판별
    """
    if not picked_sku:
        return {"consumption": None, "reservation_1y": None, "reservation_3y":None}

    cur = conn.cursor()
    cols = [r[1] for r in cur.execute(f"PRAGMA table_info('{table_name}')").fetchall()]
    has = lambda c: c in cols

    sku_col = "SkuName" if has("SkuName") else ("SKUName" if has("SKUName") else "skuName")
    arm_col = "ArmSkuName" if has("ArmSkuName") else ("ArmSku" if has("ArmSku") else None)
    prod_col = "ProductName" if has("ProductName") else None
    price_col = "RetailPrice" if has("RetailPrice") else ("retailPrice" if has("retailPrice") else "retailprice")
    pricetype_col = "PriceType" if has("PriceType") else ("priceType" if has("priceType") else None)
    uom_col = "UnitOfMeasure" if has("UnitOfMeasure") else ("unitOfMeasure" if has("unitOfMeasure") else "unit_of_measure")

    # 기간 컬럼 후보(있을 수도 있고 없을 수도 있음)
    term_cols = [c for c in ["Term", "ReservationTerm", "TermLength", "TermName"] if has(c)]

    # ---- 헬퍼: 한 줄을 dict로 변환
    def _row_to_dict(row):
        if not row: return None
        dcols = [d[0] for d in cur.description]
        return dict(zip(dcols, row))

    # ---- 헬퍼: PriceType = 'Consumption' 한 줄
    def _fetch_consumption():
        if not pricetype_col: 
            return None
        q = f'SELECT * FROM "{table_name}" WHERE {sku_col}=? AND {pricetype_col}=? LIMIT 1'
        cur.execute(q, (picked_sku, "Consumption"))
        r = _row_to_dict(cur.fetchone())
        if r: return r
        # SkuName LIKE 완화
        q = f'SELECT * FROM "{table_name}" WHERE {sku_col} LIKE ? AND {pricetype_col}=? LIMIT 1'
        cur.execute(q, (f"%{picked_sku}%", "Consumption"))
        return _row_to_dict(cur.fetchone())

    # ---- 헬퍼: PriceType/Term 다양한 표기→ 1Y/3Y 찾기
    PT_1Y = ["Reservation 1 Year", "Reserved 1 Year", "RI 1 Year", "1 Year Reserved"]
    PT_3Y = ["Reservation 3 Years", "Reservation 3 Years", "Reserved 3 Year", "Reserved 3 Years", "RI 3 Year", "RI 3 Years", "3 Years Reserved"]
    TERM_1Y = ["1 Year", "1 Years", "1Y", "12 Months", "12 Month"]
    TERM_3Y = ["3 Years", "3 Years", "3Y", "36 Months", "36 Month"]

    def _fetch_by_pricetype(targets, sku_target):
        # PriceType 자체에 연수 포함되는 케이스
        if not pricetype_col:
            return None
        q = f'SELECT * FROM "{table_name}" WHERE {sku_target} AND {pricetype_col} IN ({",".join(["?"]*len(targets))}) ORDER BY {price_col} ASC LIMIT 1'
        return _row_to_dict(cur.execute(q, targets).fetchone())

    def _sku_predicates(val):
        preds = []
        params = []
        # 1) 정확
        preds.append(f'{sku_col} = ?'); params.append(val)
        # 2) LIKE
        preds.append(f'{sku_col} LIKE ?'); params.append(f"%{val}%")
        # 3) ArmSku LIKE
        if arm_col:
            preds.append(f'{arm_col} LIKE ?'); params.append(f"%{val}%")
        return " OR ".join(f"({p})" for p in preds), params

    def _fetch_reservation_by_term(years: str):
        # PriceType='Reservation' + Term 컬럼의 값으로 1Y/3Y 판단
        if not pricetype_col or not term_cols:
            return None
        wanted_terms = TERM_1Y if years == "1" else TERM_3Y
        sku_pred, sku_params = _sku_predicates(picked_sku)

        # 여러 term 컬럼 중 하나라도 매칭되면 OK
        term_like = " OR ".join([f"{c} IN ({','.join(['?']*len(wanted_terms))})" for c in term_cols])
        q = f'''SELECT * FROM "{table_name}"
                WHERE ({sku_pred})
                  AND {pricetype_col} IN ('Reservation','Reserved','Reservation Linux','Reservation Windows','Reserved Instance','RI')
                  AND ({term_like})
                ORDER BY {price_col} ASC LIMIT 1'''
        params = sku_params + wanted_terms * len(term_cols)
        cur.execute(q, params)
        return _row_to_dict(cur.fetchone())

    def _fetch_reservation_any(years: str):
        # 1) PriceType에 직접 1Y/3Y가 들어있는 경우부터
        sku_pred, sku_params = _sku_predicates(picked_sku)
        targets = PT_1Y if years == "1" else PT_3Y
        if pricetype_col:
            q = f'''SELECT * FROM "{table_name}"
                    WHERE ({sku_pred})
                      AND {pricetype_col} IN ({",".join(["?"]*len(targets))})
                    ORDER BY {price_col} ASC LIMIT 1'''
            cur.execute(q, sku_params + targets)
            r = _row_to_dict(cur.fetchone())
            if r: return r

        # 2) PriceType='Reservation' + Term 매칭
        r = _fetch_reservation_by_term(years)
        if r: return r

        # 3) ProductName 근사(소비 가격의 ProductName과 같은 상품 중 예약 타입)
        if prod_col and pricetype_col:
            cur.execute(
                f'''SELECT {prod_col} FROM "{table_name}" 
                    WHERE {sku_col}=? AND {pricetype_col}='Consumption' LIMIT 1''',
                (picked_sku,)
            )
            pr = cur.fetchone()
            if pr:
                prod = pr[0]
                # (a) PriceType 직접 표기형
                q = f'''SELECT * FROM "{table_name}"
                        WHERE {prod_col}=? AND {pricetype_col} IN ({",".join(["?"]*len(targets))})
                        ORDER BY {price_col} ASC LIMIT 1'''
                cur.execute(q, (prod, *targets))
                r = _row_to_dict(cur.fetchone())
                if r: return r
                # (b) Reservation + Term
                if term_cols:
                    term_like = " OR ".join([f"{c} IN ({','.join(['?']*len(TERM_1Y if years=='1' else TERM_3Y))})" for c in term_cols])
                    q = f'''SELECT * FROM "{table_name}"
                            WHERE {prod_col}=?
                              AND {pricetype_col} IN ('Reservation','Reserved','Reservation Linux','Reservation Windows','Reserved Instance','RI')
                              AND ({term_like})
                            ORDER BY {price_col} ASC LIMIT 1'''
                    cur.execute(q, (prod, *(TERM_1Y if years=="1" else TERM_3Y)*len(term_cols)))
                    r = _row_to_dict(cur.fetchone())
                    if r: return r
        return None

    # ---- 실행
    cons = _fetch_consumption()
    r1y  = _fetch_reservation_any("1")
    r3y  = _fetch_reservation_any("3")

    return {"consumption": cons, "reservation_1y": r1y, "reservation_3y": r3y}

def enrich_with_heuristics(parsed: Dict[str, Any], question: str) -> Dict[str, Any]:
    """LLM 파싱이 비었거나 불완전할 때 키워드로 보정."""
    parsed = dict(parsed or {})
    t = (question or "").lower()

    def has(*keys): 
        return any(re.search(k, t, re.I) for k in keys)

    # 이미 service_type 있으면 부족한 세부만 채움
    st = (parsed.get("service_type") or "").lower()

    # --- DB 계열 ---
    if not st and has(r"\bsql\b", r"azure sql"):
        parsed["service_type"] = "DB"
        # 'sql'만 온 경우 기본을 Azure SQL Database로
        parsed.setdefault("db_type", "SQL Database")
        st = "db"
    if not st and has("mysql"):
        parsed["service_type"] = "DB"
        parsed.setdefault("db_type", "Azure Database for MySQL")
        st = "db"
    if not st and has("postgre", "postgres"):
        parsed["service_type"] = "DB"
        parsed.setdefault("db_type", "Azure Database for PostgreSQL")
        st = "db"

    # DB인데 db_type이 비어 있으면 보완
    if st == "db" and not parsed.get("db_type"):
        if has("mysql"):
            parsed["db_type"] = "Azure Database for MySQL"
        elif has("postgre", "postgres"):
            parsed["db_type"] = "Azure Database for PostgreSQL"
        else:
            parsed["db_type"] = "SQL Database"  # 기본값

    # --- Databricks ---
    if not st and has("databricks"):
        parsed["service_type"] = "Databricks"
        # 시간 힌트
        hrs = parse_hours(question, 0)
        if hrs:
            parsed["usage_hours"] = hrs
        st = "databricks"

    # --- AOAI ---
    if not st and has("aoai", "openai", r"\bgpt\b", r"gpt-?4o", r"4o"):
        parsed["service_type"] = "AOAI"
        # 모델 추정
        if has(r"gpt-?4o-?mini"):
            parsed["model"] = "gpt-4o-mini"
        elif has(r"gpt-?4o", r"\b4o\b"):
            parsed["model"] = "gpt-4o"
        st = "aoai"

        # IB/OB 숫자도 같이 뽑아보기
        txt = t.replace(",", "").replace(" ", "")
        m_ib = re.search(r'(?:ib|입력|input)(\d+)', txt)
        m_ob = re.search(r'(?:ob|출력|output)(\d+)', txt)
        nums = re.findall(r'\d+', txt)
        if m_ib: parsed["tokens_in"] = int(m_ib.group(1))
        if m_ob: parsed["tokens_out"] = int(m_ob.group(1))
        if ("tokens_in" not in parsed or "tokens_out" not in parsed) and len(nums) >= 2:
            parsed.setdefault("tokens_in", int(nums[0]))
            parsed.setdefault("tokens_out", int(nums[1]))

    return parsed


# ---- 1. Azure 문의 ----
def azure_inquiry_node(state: GraphState) -> GraphState:
    mcp = state["mcp"]
    question = mcp["payload"]["question"]

    try:
        docs = retriever.invoke(question) or []
        context = "\n\n".join(doc.page_content for doc in docs)
    except Exception as e:
        logging.exception(f"Chroma 검색 오류: {e}")
        context = ""

    prompt = f"다음 문맥을 바탕으로 Azure 질문에 한국어로 간결하게 답하세요.\n\n문맥:\n{context}\n\n질문: {question}\n\n답변:"
    answer = azure_llm.invoke(prompt).content
    mcp["payload"]["answer"] = answer
    return {"mcp": mcp, "last_agent": "azure_inquiry_agent"}

# ---- 2. 사이트 사용법 ----
def site_inquiry_node(state: GraphState) -> GraphState:
    mcp = state["mcp"]
    question = mcp["payload"]["question"]

    try:
        docs = retriever.invoke(question) or []
        context = "\n\n".join(doc.page_content for doc in docs)
    except Exception as e:
        logging.exception(f"Chroma 검색 오류: {e}")
        context = ""

    prompt = f"다음 사이트 매뉴얼 문맥을 바탕으로 사용법 질문에 간결히 답하세요.\n\n문맥:\n{context}\n\n질문: {question}\n\n답변:"
    answer = azure_llm.invoke(prompt).content
    mcp["payload"]["answer"] = answer
    return {"mcp": mcp, "last_agent": "site_inquiry_agent"}

# ---- LLM 파서 ----
def parse_user_query_with_llm(question: str) -> Dict[str, Any]:
    system_prompt = """
You extract Azure service specs from a user's query.
Return JSON ONLY. Keys: service_type in ['VM','DB','Disk','AOAI','Databricks'], cores, ram, quantity, os, disk_type, storage_gb, db_type, usage_hours, plan.
If not sure, return {}.
Examples:
- "VM 4코어 16GB 2대, 리눅스" -> {"service_type":"VM","cores":4,"ram":16,"quantity":2,"os":"Linux"}
- "Azure SQL 3600시간" -> {"service_type":"DB","db_type":"Azure SQL","usage_hours":3600}
- "SSD 500GB 3개" -> {"service_type":"Disk","disk_type":"SSD","storage_gb":500,"quantity":3}
- "gpt-4o 100만 토큰 입력, 50만 토큰 출력" -> {"service_type":"AOAI","model":"gpt-4o","tokens_in":1000000,"tokens_out":500000}
"""
    try:
        msgs = [SystemMessage(content=system_prompt), HumanMessage(content=question)]
        response = azure_llm.invoke(msgs)
        txt = (response.content or "").strip()
        data = json.loads(txt)
    except Exception:
        data = {}

    # LLM 결과 보강(안전망)
    return enrich_with_heuristics(data, question)

# ---- spec_matching_node 위쪽 또는 그 근처에 추가 ----
def pick_disk_by_capacity(conn, table_name: str, disk_type_hint: str, requested_gb: int):
    """
    disk_type_hint: 'ssd' or 'hdd' (lower)
    requested_gb: 요구 용량(GB)
    규칙:
      - SSD -> DiskCategory LIKE '%Standard SSD%' (간단 규칙)
      - HDD -> DiskCategory LIKE '%Standard HDD%'
      - Provisioned_GiB >= requested_gb 중 가장 작은 것
      - PriceType='Consumption' 1건
    """
    cur = conn.cursor()
    cols = [r[1] for r in cur.execute(f"PRAGMA table_info('{table_name}')").fetchall()]
    has = lambda c: c in cols
    if not all(has(c) for c in ["DiskCategory", "Provisioned_GiB"]):
        return None

    cat_like = "%Standard SSD%" if "ssd" in (disk_type_hint or "").lower() else "%Standard HDD%"
    q = f'''
    SELECT * FROM "{table_name}"
    WHERE PriceType='Consumption'
      AND DiskCategory LIKE ?
      AND Provisioned_GiB >= ?
    ORDER BY Provisioned_GiB ASC
    LIMIT 1
    '''
    cur.execute(q, (cat_like, int(requested_gb)))
    row = cur.fetchone()
    if not row: 
        # 없으면 용량 조건만 빼고 가장 작은 카테고리 하나라도
        q2 = f'''
        SELECT * FROM "{table_name}"
        WHERE PriceType='Consumption' AND DiskCategory LIKE ?
        ORDER BY Provisioned_GiB ASC
        LIMIT 1
        '''
        cur.execute(q2, (cat_like,))
        row = cur.fetchone()
        if not row:
            return None
    cols = [d[0] for d in cur.description]
    return dict(zip(cols, row))


def pick_db_one_row(conn, table_name: str, db_type: str):
    """
    db_type: 'sql', 'mysql', 'postgresql' 등 추정 키워드
    PriceType='Consumption' 한 건 반환
    ProductName/MeterName/SkuName 등에서 키워드 근사 매칭
    """
    cur = conn.cursor()
    cols = [r[1] for r in cur.execute(f"PRAGMA table_info('{table_name}')").fetchall()]
    has = lambda c: c in cols

    # 후보 컬럼
    prod = "ProductName" if has("ProductName") else None
    meter = "MeterName" if has("MeterName") else None
    sku   = "SkuName" if has("SkuName") else ( "SKUName" if has("SKUName") else None )

    # 키워드 맵(간단)
    if "mysql" in db_type.lower():
        keys = ["mysql"]
    elif "postgre" in db_type.lower():
        keys = ["postgres", "postgre", "postgresql"]
    else:
        # default: Azure SQL Database
        keys = ["sql database", "azure sql"]

    where_parts = ["PriceType='Consumption'"]
    params = []
    like_parts = []
    for k in keys:
        chunk = []
        if prod:  chunk.append(f"{prod} LIKE ?")
        if meter: chunk.append(f"{meter} LIKE ?")
        if sku:   chunk.append(f"{sku} LIKE ?")
        like_parts.append("(" + " OR ".join(chunk) + ")")
        params += [f"%{k}%"] * len(chunk)

    if like_parts:
        where = " AND (" + " OR ".join(like_parts) + ")"
    else:
        where = ""

    q = f'SELECT * FROM "{table_name}" WHERE ' + " AND ".join(["PriceType='Consumption'"]) + where + " LIMIT 1"
    cur.execute(q, params)
    row = cur.fetchone()
    if not row:
        return None
    cols = [d[0] for d in cur.description]
    return dict(zip(cols, row))


# ---- 3-1. 스펙 매칭 ----
def spec_matching_node(state: GraphState) -> GraphState:
    mcp = state["mcp"]
    question = mcp["payload"]["question"]
    session_id = mcp["payload"]["metadata"]["session_id"]
    parsed = parse_user_query_with_llm(question)
    parsed = enrich_with_heuristics(parsed, question)

    # 서비스 타입 없으면 가이드
    if not parsed.get("service_type"):
        guide_all = "요구 스펙을 더 알려주세요.\n\n" + "\n".join([f"[{k.upper()}]\n{v}" for k, v in SERVICE_GUIDE.items()])
        mcp["payload"]["answer"] = guide_all
        return {"mcp": mcp, "last_agent": "spec_matching_agent"}

    svc_type = str(parsed["service_type"]).lower()

    def need_guide(st: str) -> str | None:
        if st == "vm":
            if not parsed.get("cores") or not parsed.get("ram") or not parsed.get("os"):
                return SERVICE_GUIDE["vm"]
        if st == "disk":
            if not parsed.get("disk_type"):
                parsed["disk_type"] = "ssd" if re.search(r"ssd", question, re.I) else ("hdd" if re.search(r"hdd", question, re.I) else None)
            if not parsed.get("storage_gb"):
                m = re.search(r'(\d{2,6})\s*gb', question, re.I)
                if m: parsed["storage_gb"] = int(m.group(1))
            if not parsed.get("disk_type") or not parsed.get("storage_gb"):
                return SERVICE_GUIDE["disk"]
        if st == "db":
            if not parsed.get("db_type") and not re.search(r"(mysql|maria|postgre|sql)", question, re.I):
                return SERVICE_GUIDE["db"]
        if st == "databricks":
            # 시간은 이 노드에서 안 묻고 기본 720h로 처리할 예정
            pass
        if st == "aoai":
            if not parsed.get("model"):
                return SERVICE_GUIDE["aoai"] + "\n\n예: gpt-4o / gpt-4o-mini 중 선택"
        return None

    g = need_guide(svc_type)
    if g:
        mcp["payload"]["answer"] = g
        return {"mcp": mcp, "last_agent": "spec_matching_agent"}

    table_map = {
        "vm": "azure_vm_prices_filtered_specs_koreacentral",
        "disk": "azure_disk_prices_filtered_specs_koreacentral",
        "db": "filtered_db_compute_prices",
        "databricks": "0821_databricks_prices_koreacentral",
        "aoai": "0821_aoai_prices_koreacentral",
    }
    table_name = table_map.get(svc_type)
    if not table_name:
        mcp["payload"]["answer"] = f"'{svc_type}'용 테이블을 찾지 못했습니다."
        return {"mcp": mcp, "last_agent": "spec_matching_agent"}

    try:
        conn = sqlite3.connect(DATABASE_NAME)
        cur = conn.cursor()
        cols = [r[1] for r in cur.execute(f'PRAGMA table_info("{table_name}")').fetchall()]
        has = lambda c: c in cols

        # ---------- VM ----------
        if svc_type == "vm":
            ram_col = "MemoryGB" if has("MemoryGB") else ("RAMGB" if has("RAMGB") else None)
            where, params = [], []
            if has("PriceType"): where.append("PriceType='Consumption'")
            if parsed.get("cores") is not None and has("Cores"):
                where.append("Cores=?"); params.append(int(parsed["cores"]))
            if parsed.get("ram") is not None and ram_col:
                where.append(f"{ram_col}=?"); params.append(int(parsed["ram"]))
            if parsed.get("os") and has("OperatingSystem"):
                where.append("OperatingSystem LIKE ?"); params.append(f"%{parsed['os']}%")

            q = f'SELECT * FROM "{table_name}" WHERE ' + (" AND ".join(where) if where else "1=1") + " LIMIT 1"
            cur.execute(q, params)
            row = cur.fetchone()
            if not row:
                mcp["payload"]["answer"] = "요청 스펙에 근접한 VM SKU를 찾지 못했습니다. 코어/메모리/OS를 조정해 주세요."
                return {"mcp": mcp, "last_agent": "spec_matching_agent"}

            cns = [d[0] for d in cur.description]
            rec = dict(zip(cns, row))
            sku = rec.get("SkuName") or rec.get("SKUName") or rec.get("skuName")
            svc = rec.get("ProductName") or "VM"
            qty = int(parsed.get("quantity", 1))

            tri = fetch_three_prices_for_sku(conn, table_name, sku)
            def price_cell(d): return float((d or {}).get("RetailPrice") or 0.0)
            def uom_cell(d):   return ((d or {}).get("UnitOfMeasure") or "1 Hour").strip()
            p_cons = price_cell(tri["consumption"]); u_cons = uom_cell(tri["consumption"])
            p_r1y  = price_cell(tri["reservation_1y"]); u_r1y  = uom_cell(tri["reservation_1y"])
            p_r3y  = price_cell(tri["reservation_3y"]); u_r3y  = uom_cell(tri["reservation_3y"])

            msg = "\n".join([
                f"**선택된 SKU:** {sku}",
                f"- Consumption: {both_prices_line(p_cons, u_cons)}",
                f"- Reservation 1Y: {both_prices_line(p_r1y, u_r1y) if p_r1y else 'N/A'}",
                f"- Reservation 3Y: {both_prices_line(p_r3y, u_r3y) if p_r3y else 'N/A'}",
                "",
                "원하시는 요금제를 선택해 주세요. (예: **종량제**, **예약제 1Y**, **예약제 3Y**)"
            ])
            pending = {
                "awaiting": "plan", "service_type": "vm", "table": table_name,
                "sku_name": sku, "product_name": svc, "quantity": qty,
                "prices": {
                    "consumption": {"usd": p_cons, "uom": u_cons},
                    "reservation_1y": {"usd": p_r1y, "uom": u_r1y} if p_r1y else None,
                    "reservation_3y": {"usd": p_r3y, "uom": u_r3y} if p_r3y else None,
                }
            }
            pending_by_session_id[session_id] = pending
            mcp["payload"]["pending"] = pending
            mcp["payload"]["awaiting"] = "plan"
            mcp["payload"]["recommended_item"] = None
            mcp["payload"]["answer"] = msg
            return {"mcp": mcp, "last_agent": "spec_matching_agent"}

        # ---------- DISK (묻지 않고 1개월로 즉시 견적) ----------
        if svc_type == "disk":
            want_cat = normalize_disk_category(parsed.get("disk_type", "ssd"))
            want_gb = int(parsed.get("storage_gb", 0))
            if want_gb <= 0:
                m = re.search(r'(\d{2,6})\s*gb', question, re.I)
                if m: want_gb = int(m.group(1))
            if want_gb <= 0:
                mcp["payload"]["answer"] = SERVICE_GUIDE["disk"]
                return {"mcp": mcp, "last_agent": "spec_matching_agent"}

            if not (has("Provisioned_GiB") and has("DiskCategory")):
                mcp["payload"]["answer"] = "디스크 테이블 스키마에 Provisioned_GiB/DiskCategory가 없습니다."
                return {"mcp": mcp, "last_agent": "spec_matching_agent"}

            price_col = "RetailPrice" if has("RetailPrice") else "retailPrice"
            uom_col = "UnitOfMeasure" if has("UnitOfMeasure") else "unitOfMeasure"

            where = ["DiskCategory LIKE ?", "CAST(Provisioned_GiB AS INTEGER) >= ?"]
            params = [f"%{want_cat}%", want_gb]
            if has("PriceType"):
                where.append("PriceType='Consumption'")

            q = f'''
            SELECT * FROM "{table_name}"
            WHERE {' AND '.join(where)}
            ORDER BY CAST(Provisioned_GiB AS INTEGER) ASC
            LIMIT 1
            '''
            cur.execute(q, params)
            row = cur.fetchone()
            if not row:
                mcp["payload"]["answer"] = "해당 조건(디스크 종류/용량)에 맞는 SKU를 찾지 못했습니다. 용량을 조금 늘려보세요."
                return {"mcp": mcp, "last_agent": "spec_matching_agent"}

            cns = [d[0] for d in cur.description]
            rec = dict(zip(cns, row))
            sku = rec.get("SkuName") or rec.get("MeterName") or rec.get("ProductName") or "Managed Disk"
            usd = float(rec.get(price_col) or 0.0)
            uom = (rec.get(uom_col) or "1 Month").strip() or "1 Month"
            qty = int(parsed.get("quantity", 1))
            prov = int(rec.get("Provisioned_GiB") or want_gb)

            # 기본 1개월로 즉시 계산
            months = DEFAULT_MONTHS
            if "gb" in uom.lower() and "month" in uom.lower():
                total_usd = usd * prov * months * qty
            else:
                total_usd = usd * months * qty
            total_krw = to_krw(total_usd)

            mcp["payload"]["cart_ctx"] = {
                "kind": "disk", "usd_per_unit": usd, "months": months, "storage_gb": prov,
                "mode": "gb_month" if ("gb" in uom.lower() and "month" in uom.lower()) else "month",
            }
            spec_text = f"{sku} / Storage / {prov}GB / {months}개월"
            spec_line = (
                f"**선택된 스토리지 SKU:** {sku}\n"
                f"- 카테고리: {rec.get('DiskCategory','N/A')}\n"
                f"- 프로비저닝 용량: {prov} GiB\n"
                f"- 단가: {both_prices_line(usd, uom)}"
            )

            mcp["payload"]["recommended_item"] = {
                "service": rec.get("ProductName") or "Managed Disk",
                "option": f"{sku} / {rec.get('DiskCategory','')} / {prov}GB / {months}개월",
                "quantity": qty,
                "price": fmt_money_krw(total_krw),
                "spec": f"{sku} | {rec.get('DiskCategory','')} | {prov}GB"
            }
            mcp["payload"]["awaiting"] = "cart_quantity"
            mcp["payload"]["pending"] = None
            pending_by_session_id.pop(session_id, None)
            mcp["payload"]["answer"] = (
                f"{spec_line}\n\n"
                f"**스토리지 1차 견적(기본 {months}개월, 현재 {qty}개)**\n"
                f"- 합계: {fmt_usd(total_usd)} ≈ {fmt_money_krw(total_krw)}\n\n"
                f"장바구니 추가 전, **수량을 몇 개로 할까요?** (예: 5)"
            )
            return {"mcp": mcp, "last_agent": "spec_matching_agent"}

        # ---------- DB ----------
        if svc_type == "db":
            want_raw = (parsed.get("db_type") or question or "").lower()
            if "mysql" in want_raw:
                want_key = "azure database for mysql"
            elif "postgre" in want_raw:
                want_key = "azure database for postgre"
            else:
                want_key = "sql database"

            tri = fetch_db_three_prices_by_service(conn, "filtered_db_compute_prices", want_key)

            def _fmt_line(x):
                return both_prices_line(x["usd"], x["uom"]) if (x and x.get("usd")) else "N/A"

            disp_sku = None
            for k in ("consumption","reservation_1y","reservation_3y"):
                if tri.get(k) and tri[k].get("sku"):
                    disp_sku = tri[k]["sku"]; break
            if not disp_sku:
                disp_sku = want_key.upper()

            msg = "\n".join([
                f"**선택된 DB SKU:** {disp_sku}",
                f"- Consumption: {_fmt_line(tri['consumption'])}",
                f"- Reservation 1Y: {_fmt_line(tri['reservation_1y'])}",
                f"- Reservation 3Y: {_fmt_line(tri['reservation_3y'])}",
                "",
                "원하시는 요금제를 선택해 주세요. (예: **종량제**, **예약제 1Y**, **예약제 3Y**)"
            ])

            prices = {
                "consumption": tri["consumption"] and {"usd": tri["consumption"]["usd"], "uom": tri["consumption"]["uom"]} or None,
                "reservation_1y": tri["reservation_1y"] and {"usd": tri["reservation_1y"]["usd"], "uom": tri["reservation_1y"]["uom"]} or None,
                "reservation_3y": tri["reservation_3y"] and {"usd": tri["reservation_3y"]["usd"], "uom": tri["reservation_3y"]["uom"]} or None,
            }

            pending = {
                "awaiting": "plan",
                "service_type": "db",
                "table": "filtered_db_compute_prices",
                "sku_name": disp_sku,
                "product_name": want_key.title(),
                "quantity": int(parsed.get("quantity", 1)),
                "prices": prices
            }
            pending_by_session_id[session_id] = pending
            mcp["payload"]["pending"] = pending
            mcp["payload"]["awaiting"] = "plan"
            mcp["payload"]["recommended_item"] = None
            mcp["payload"]["answer"] = msg
            return {"mcp": mcp, "last_agent": "spec_matching_agent"}

        # ---------- Databricks (유형만 받으면 720시간으로 즉시 계산) ----------
        if svc_type == "databricks":
            tiers = fetch_databricks_tiers(conn, table_name)  # {"standard": {...}, "premium": {...}}
            std = tiers.get("standard"); prm = tiers.get("premium")
            if not (std or prm):
                mcp["payload"]["answer"] = "Databricks SKU를 찾지 못했습니다. (Standard/Premium DBU)"
                return {"mcp": mcp, "last_agent": "spec_matching_agent"}

            # 사용자 입력에서 plan 추출
            plan_txt = (parsed.get("plan") or "").lower()
            if not plan_txt:
                m_plan = re.search(r"(standard|premium)", question, re.I)
                plan_txt = (m_plan.group(1).lower() if m_plan else "")

            if plan_txt in ("standard", "premium"):
                chosen = std if plan_txt == "standard" else prm
                usd_per_h = float(chosen["RetailPrice"])
                hours = DEFAULT_MONTH_HOURS
                qty = int(parsed.get("quantity", 1))
                total_usd = usd_per_h * hours * qty
                total_krw = to_krw(total_usd)
                mcp["payload"]["cart_ctx"] = {"kind": "consumption", "usd_per_hour": usd_per_h, "hours": hours}
                mcp["payload"]["recommended_item"] = {
                    "service": "Databricks",
                    "option": f"All-purpose Compute DBU / {plan_txt.capitalize()} / {hours}시간",
                    "quantity": qty,
                    "price": fmt_money_krw(total_krw),
                }
                mcp["payload"]["awaiting"] = "cart_quantity"
                mcp["payload"]["pending"] = None
                pending_by_session_id.pop(session_id, None)
                mcp["payload"]["answer"] = (
                    f"**Databricks {plan_txt.capitalize()} 1차 견적(기본 {hours}시간, 현재 {qty}개)**\n"
                    f"- 합계: {fmt_usd(total_usd)} ≈ {fmt_money_krw(total_krw)}\n\n"
                    f"장바구니 추가 전, **수량을 몇 개로 할까요?** (예: 5)"
                )
                return {"mcp": mcp, "last_agent": "spec_matching_agent"}

            # plan이 없으면 둘 다 보여주고 plan만 선택받음(시간은 720h로 자동)
            def unit_str(rec):
                if not rec: return "N/A"
                return both_prices_line(float(rec["RetailPrice"]), rec.get("UnitOfMeasure") or "1 Hour")
            pending = {
                "awaiting": "dbx_plan",
                "service_type": "databricks",
                "table": table_name,
                "product_name": "Databricks",
                "quantity": int(parsed.get("quantity", 1)),
                "hours": DEFAULT_MONTH_HOURS,  # 기본값 미리 세팅
                "prices": {
                    "standard": {"usd_per_hour": float(std["RetailPrice"]), "uom": std.get("UnitOfMeasure") or "1 Hour"} if std else None,
                    "premium":  {"usd_per_hour": float(prm["RetailPrice"]), "uom": prm.get("UnitOfMeasure") or "1 Hour"} if prm else None,
                }
            }
            pending_by_session_id[session_id] = pending
            mcp["payload"]["pending"] = pending
            mcp["payload"]["awaiting"] = "dbx_plan"
            mcp["payload"]["answer"] = (
                "**Databricks DBU 단가**\n"
                f"- Standard: {unit_str(std)}\n"
                f"- Premium:  {unit_str(prm)}\n\n"
                "원하시는 **유형**을 선택해 주세요. (시간은 기본 720시간으로 계산합니다)"
            )
            return {"mcp": mcp, "last_agent": "spec_matching_agent"}

        # ---------------- AOAI (변경 없음: 토큰은 시간/개월과 무관) ----------------
        # ---- spec_matching_node AOAI 분기 (수정된 전문) ----
        # ---------------- AOAI (무조건 UI부터 보여주도록 개선) ----------------
        if svc_type == "aoai":
            # 모델 추출
            raw = (parsed.get("model") or question or "").lower()
            t = re.sub(r"[\s_]+", "-", raw)
            if re.search(r"gpt-?4o-?mini", t):
                model_key = "gpt-4o-mini"
            elif re.search(r"gpt-?4o", t) or "4o" in t:
                model_key = "gpt-4o"
            else:
                model_key = "gpt-4o"

            io = fetch_aoai_inout_prices(conn, table_name, model_key)
            inp, out = io.get("input"), io.get("output")
            if not (inp and out):
                mcp["payload"]["answer"] = f"{model_key} 단가를 찾지 못했습니다. (Input/Output Tokens)"
                return {"mcp": mcp, "last_agent": "spec_matching_agent"}

            price_in = {
                "usd": float(inp.get("retailprice") or inp.get("RetailPrice") or 0.0),
                "uom": inp.get("unitofmeasure") or inp.get("UnitOfMeasure") or "1K tokens"
            }
            price_out = {
                "usd": float(out.get("retailprice") or out.get("RetailPrice") or 0.0),
                "uom": out.get("unitofmeasure") or out.get("UnitOfMeasure") or "1K tokens"
            }

            qty = int(parsed.get("quantity", 1))

            # ---- 무조건 토큰 입력을 먼저 요청 ----
            pending = {
                "awaiting": "tokens",
                "service_type": "aoai",
                "table": table_name,
                "sku_name": model_key,
                "product_name": model_key,
                "quantity": qty,
                "prices": {"input": price_in, "output": price_out}
            }
            pending_by_session_id[session_id] = pending
            mcp["payload"]["pending"] = pending
            mcp["payload"]["awaiting"] = "tokens"

            # 텍스트 안내 + UI 동시 제공
            mcp["payload"]["answer"] = (
                f"**Azure OpenAI(AOAI) 견적 산출**\n"
                f"- Input 단가: {both_prices_line(price_in['usd'], price_in['uom'])}\n"
                f"- Output 단가: {both_prices_line(price_out['usd'], price_out['uom'])}\n\n"
                "모델과 입력/출력 토큰 수를 선택해 주세요."
            )
            mcp["payload"]["ui"] = {
                "type": "aoai_form",
                "options": ["gpt-4o", "gpt-4o-mini"],  # 빠른 선택 버튼
                "inputs": [
                    {"label": "입력 토큰 수 (IB)", "name": "tokens_in", "type": "number", "min": 1000, "step": 1000},
                    {"label": "출력 토큰 수 (OB)", "name": "tokens_out", "type": "number", "min": 1000, "step": 1000}
                ],
                "submit_label": "견적 산출"
            }
            return {"mcp": mcp, "last_agent": "spec_matching_agent"}

            
    except Exception:
        logging.exception("DB 조회 오류")
        mcp["payload"]["answer"] = "데이터베이스 조회 중 오류가 발생했습니다."
        return {"mcp": mcp, "last_agent": "spec_matching_agent"}
    finally:
        try: conn.close()
        except: pass


# ----- [업데이트] 요금제 선택 & (종량제/개월/토큰 입력 처리) -----
def plan_or_hours_node(state: GraphState) -> GraphState:
    mcp = state["mcp"]
    q = (mcp["payload"]["question"] or "").strip()
    sid = mcp["payload"]["metadata"]["session_id"]

    pending = (
        mcp["payload"].get("pending")
        or pending_by_session_id.get(sid)
        or (SESSION_PAYLOADS.get(sid, {}) if 'SESSION_PAYLOADS' in globals() else {}).get("pending")
        or {}
    )
    if pending:
        mcp["payload"]["pending"] = pending

    if not pending:
        mcp["payload"]["answer"] = "진행 중인 견적 맥락이 없습니다. 스펙을 먼저 알려주세요. (예: Databricks Standard / VM 4코어 16GB 등)"
        return {"mcp": mcp, "last_agent": "plan_or_hours_node"}

    svc_type = pending.get("service_type", "")
    awaiting = pending.get("awaiting")

    # ============================================================
    # (A) VM/DB — 플랜 선택 후 시간은 묻지 않고 720h로 고정
    # ============================================================
    if awaiting == "plan" and svc_type in ("vm", "db"):
        txt = q.lower()
        qty = int(pending.get("quantity", 1))

        # 종량제 → 720h 기본
        if any(k in txt for k in ["종량제", "consumption"]):
            p = pending["prices"].get("consumption")
            if not p or not p.get("usd"):
                mcp["payload"]["answer"] = "종량제 가격을 찾지 못했습니다."
                return {"mcp": mcp, "last_agent": "plan_or_hours_node"}
            usd_per_hour = float(p.get("usd", 0.0))
            hours = DEFAULT_MONTH_HOURS
            total_usd = usd_per_hour * hours * qty
            total_krw = to_krw(total_usd)

            mcp["payload"]["cart_ctx"] = {"kind": "consumption", "usd_per_hour": usd_per_hour, "hours": hours}
            mcp["payload"]["recommended_item"] = {
                "service": pending["product_name"],
                "option": f"{pending.get('sku_name') or pending['product_name']} / Consumption / {hours}시간",
                "quantity": qty,
                "price": fmt_money_krw(total_krw),
            }
            pending_by_session_id.pop(sid, None)
            mcp["payload"]["pending"] = None
            mcp["payload"]["awaiting"] = "cart_quantity"
            mcp["payload"]["answer"] = (
                f"**종량제 1차 견적(기본 {hours}시간, 현재 {qty}개)**\n"
                f"- 합계: {fmt_usd(total_usd)} ≈ {fmt_money_krw(total_krw)}\n\n"
                f"장바구니 추가 전, **수량을 몇 개로 할까요?** (예: 5)"
            )
            return {"mcp": mcp, "last_agent": "plan_or_hours_node"}

        # 예약제 1Y
        if any(k in txt for k in ["예약제 1y", "예약제1y", "1y", "1년", "reservation 1y"]):
            p = pending["prices"].get("reservation_1y")
            if not p:
                mcp["payload"]["answer"] = "해당 SKU의 예약제 1Y 가격을 찾지 못했습니다. 종량제를 이용해 주세요."
                return {"mcp": mcp, "last_agent": "plan_or_hours_node"}
            qty = int(pending["quantity"])
            total_krw, total_usd, _ = compute_reservation_total(p, months=12, quantity=qty)
            mcp["payload"]["cart_ctx"] = {
                "kind": "reservation", "usd": float(p.get("usd", 0.0)),
                "uom": (p.get("uom") or "").lower(), "months": 12,
                "hours_per_month": MONTHLY_HOURS,
            }
            mcp["payload"]["recommended_item"] = {
                "service": pending["product_name"],
                "option": f"{pending.get('sku_name') or pending['product_name']} / Reservation 1Y",
                "quantity": qty, "price": fmt_money_krw(total_krw),
            }
            pending_by_session_id.pop(sid, None)
            mcp["payload"]["pending"] = None
            mcp["payload"]["awaiting"] = "cart_quantity"
            mcp["payload"]["answer"] = (
                f"**예약제 1년 1차 견적 (현재 {qty}대)**\n"
                f"- 합계: {fmt_usd(total_usd)} ≈ {fmt_money_krw(total_krw)}\n"
                f"장바구니 추가 전, **수량을 몇 대로 할까요?** (예: 5)"
            )
            return {"mcp": mcp, "last_agent": "plan_or_hours_node"}

        # 예약제 3Y
        if any(k in txt for k in ["예약제 3y", "예약제3y", "3y", "3년", "reservation 3y"]):
            p = pending["prices"].get("reservation_3y")
            if not p:
                mcp["payload"]["answer"] = "해당 SKU의 예약제 3Y 가격을 찾지 못했습니다. 종량제를 이용해 주세요."
                return {"mcp": mcp, "last_agent": "plan_or_hours_node"}
            qty = int(pending["quantity"])
            total_krw, total_usd, _ = compute_reservation_total(p, months=36, quantity=qty)
            mcp["payload"]["cart_ctx"] = {
                "kind": "reservation", "usd": float(p.get("usd", 0.0)),
                "uom": (p.get("uom") or "").lower(), "months": 36,
                "hours_per_month": MONTHLY_HOURS,
            }
            mcp["payload"]["recommended_item"] = {
                "service": pending["product_name"],
                "option": f"{pending.get('sku_name') or pending['product_name']} / Reservation 3Y",
                "quantity": qty, "price": fmt_money_krw(total_krw),
            }
            pending_by_session_id.pop(sid, None)
            mcp["payload"]["pending"] = None
            mcp["payload"]["awaiting"] = "cart_quantity"
            mcp["payload"]["answer"] = (
                f"**예약제 3년 1차 견적 (현재 {qty}대)**\n"
                f"- 합계: {fmt_usd(total_usd)} ≈ {fmt_money_krw(total_krw)}\n"
                f"장바구니 추가 전, **수량을 몇 대로 할까요?** (예: 5)"
            )
            return {"mcp": mcp, "last_agent": "plan_or_hours_node"}

        mcp["payload"]["answer"] = "요금제를 인식하지 못했습니다. **종량제 / 예약제 1Y / 예약제 3Y** 중 선택해 주세요."
        return {"mcp": mcp, "last_agent": "plan_or_hours_node"}

    # ============================================================
    # (B) Databricks — 유형만 고르면 720h 자동
    # ============================================================
    if awaiting == "dbx_plan" and svc_type == "databricks":
        txt = q.lower()
        choice = "standard" if "standard" in txt else ("premium" if "premium" in txt else None)
        if not choice:
            mcp["payload"]["answer"] = "원하는 유형을 선택해 주세요. (**Standard** / **Premium**) (시간은 기본 720h)"
            return {"mcp": mcp, "last_agent": "plan_or_hours_node"}

        price_rec = pending["prices"].get(choice)
        if not price_rec:
            mcp["payload"]["answer"] = f"{choice.capitalize()} 단가를 찾지 못했습니다."
            return {"mcp": mcp, "last_agent": "plan_or_hours_node"}

        hours = int(pending.get("hours") or DEFAULT_MONTH_HOURS)
        usd_per_hour = float(price_rec["usd_per_hour"])
        qty = int(pending.get("quantity", 1))
        total_usd = usd_per_hour * hours * qty
        total_krw = to_krw(total_usd)

        mcp["payload"]["cart_ctx"] = {"kind": "consumption", "usd_per_hour": usd_per_hour, "hours": hours}
        mcp["payload"]["recommended_item"] = {
            "service": pending["product_name"],
            "option": f"Databricks / {choice.capitalize()} / {hours}시간",
            "quantity": qty,
            "price": fmt_money_krw(total_krw),
        }
        pending_by_session_id.pop(sid, None)
        mcp["payload"]["pending"] = None
        mcp["payload"]["awaiting"] = "cart_quantity"
        mcp["payload"]["answer"] = (
            f"**Databricks {choice.capitalize()} 1차 견적(기본 {hours}시간, 현재 {qty}개)**\n"
            f"- 합계: {fmt_usd(total_usd)} ≈ {fmt_money_krw(total_krw)}\n\n"
            f"장바구니 추가 전, **수량을 몇 개로 할까요?** (예: 5)"
        )
        return {"mcp": mcp, "last_agent": "plan_or_hours_node"}

    # ============================================================
    # (C) 디스크/시간 단계는 더 이상 사용하지 않음(하위 호환: 들어오면 기본값 적용)
    # ============================================================
    if awaiting == "hours" and svc_type in ("vm", "db", "databricks"):
        # 혹시 기존 플로우로 진입해도 기본 720h로 처리
        p = pending["prices"]["consumption"]
        usd_per_hour = float(p.get("usd", 0.0))
        hours = DEFAULT_MONTH_HOURS
        qty = int(pending["quantity"])
        total_usd = usd_per_hour * hours * qty
        total_krw = to_krw(total_usd)
        mcp["payload"]["cart_ctx"] = {"kind": "consumption", "usd_per_hour": usd_per_hour, "hours": hours}
        mcp["payload"]["recommended_item"] = {
            "service": pending["product_name"],
            "option": f"{pending.get('sku_name') or pending['product_name']} / Consumption / {hours}시간",
            "quantity": qty,
            "price": fmt_money_krw(total_krw),
        }
        pending_by_session_id.pop(sid, None); mcp["payload"]["pending"] = None
        mcp["payload"]["awaiting"] = "cart_quantity"
        mcp["payload"]["answer"] = (
            f"**종량제 1차 견적(기본 {hours}시간, 현재 {qty}개)**\n"
            f"- 합계: {fmt_usd(total_usd)} ≈ {fmt_money_krw(total_krw)}\n\n"
            f"장바구니 추가 전, **수량을 몇 개로 할까요?** (예: 5)"
        )
        return {"mcp": mcp, "last_agent": "plan_or_hours_node"}

    if awaiting == "months" and svc_type == "disk":
        # 혹시 들어와도 1개월로 처리
        p = pending["prices"]["consumption"]
        usd_unit = float(p.get("usd", 0.0))
        uom = (p.get("uom") or "").lower()
        qty = int(pending["quantity"])
        gb = int(pending.get("storage_gb", 0))
        months = DEFAULT_MONTHS
        if ("gb" in uom) and ("month" in uom):
            total_usd = usd_unit * gb * months * qty
        else:
            total_usd = usd_unit * months * qty
        total_krw = to_krw(total_usd)

        mcp["payload"]["cart_ctx"] = {
            "kind": "disk", "usd_per_unit": usd_unit, "months": months, "storage_gb": gb,
            "mode": "gb_month" if ("gb" in uom and "month" in uom) else "month",
        }
        mcp["payload"]["recommended_item"] = {
            "service": pending["product_name"],
            "option": f"{pending['sku_name']} / Storage / {gb}GB / {months}개월",
            "quantity": qty,
            "price": fmt_money_krw(total_krw),
        }
        pending_by_session_id.pop(sid, None)
        mcp["payload"]["pending"] = None
        mcp["payload"]["awaiting"] = "cart_quantity"
        mcp["payload"]["answer"] = (
            f"**스토리지 1차 견적(기본 {months}개월, 현재 {qty}개)**\n"
            f"- 합계: {fmt_usd(total_usd)} ≈ {fmt_money_krw(total_krw)}\n\n"
            f"장바구니 추가 전, **수량을 몇 개로 할까요?** (예: 5)"
        )
        return {"mcp": mcp, "last_agent": "plan_or_hours_node"}

    # ============================================================
    # (D) AOAI tokens 단계 — 기존 유지
    # ============================================================
    if awaiting == "tokens" and svc_type == "aoai":
        txt = q.replace(",", "").replace(" ", "").lower()
        ib = None
        ob = None

        # "IB=1000000, OB=500000" 같은 입력 감지
        m_ib = re.search(r'(ib|입력|input)[:=]?(\d+)', txt)
        m_ob = re.search(r'(ob|출력|output)[:=]?(\d+)', txt)
        if m_ib: ib = int(m_ib.group(2))
        if m_ob: ob = int(m_ob.group(2))

        # 숫자 2개 이상 있으면 자동 매핑
        if ib is None or ob is None:
            nums = re.findall(r'\d+', txt)
            if len(nums) >= 2:
                if ib is None: ib = int(nums[0])
                if ob is None: ob = int(nums[1])

        if ib is None or ob is None:
            mcp["payload"]["answer"] = (
                "입력/출력 토큰 수를 정확히 알려주세요.\n"
                "예: `IB=1000000, OB=500000`"
            )
            return {"mcp": mcp, "last_agent": "plan_or_hours_node"}

        # 가격 정보 꺼내오기
        prices = pending.get("prices", {})
        price_in = prices.get("input")
        price_out = prices.get("output")
        if not price_in or not price_out:
            mcp["payload"]["answer"] = "단가 정보를 불러오지 못했습니다. 다시 시도해 주세요."
            return {"mcp": mcp, "last_agent": "plan_or_hours_node"}

        usd_in = float(price_in["usd"])
        usd_out = float(price_out["usd"])
        qty = int(pending.get("quantity", 1))

        usd_input = (ib / 1000.0) * usd_in
        usd_output = (ob / 1000.0) * usd_out
        total_usd = (usd_input + usd_output) * qty
        total_krw = to_krw(total_usd)

        mcp["payload"]["cart_ctx"] = {
            "kind": "aoai",
            "usd_per_1k_in": usd_in,
            "usd_per_1k_out": usd_out,
            "ib": ib,
            "ob": ob
        }
        mcp["payload"]["recommended_item"] = {
            "service": pending.get("product_name", "AOAI"),
            "option": f"{pending.get('sku_name')} / IB {ib:,} / OB {ob:,}",
            "quantity": qty,
            "price": fmt_money_krw(total_krw),
        }
        pending_by_session_id.pop(sid, None)
        mcp["payload"]["pending"] = None
        mcp["payload"]["awaiting"] = "cart_quantity"
        mcp["payload"]["answer"] = (
            f"**AOAI 1차 견적 (현재 {qty}개)**\n"
            f"- 합계: {fmt_usd(total_usd)} ≈ {fmt_money_krw(total_krw)}\n\n"
            f"장바구니 추가 전, **수량을 몇 개로 할까요?** (예: 5)"
        )
        return {"mcp": mcp, "last_agent": "plan_or_hours_node"}

# ---- [업데이트] 수량 확인 단계 → 최종 금액 재계산 & 장바구니 확인 ----
def cart_quantity_node(state: GraphState) -> GraphState:
    mcp = state["mcp"]
    q = (mcp["payload"]["question"] or "").strip()
    awaiting = mcp["payload"].get("awaiting")
    item = mcp["payload"].get("recommended_item")
    ctx = mcp["payload"].get("cart_ctx") or {}

    if awaiting != "cart_quantity" or not item or not ctx:
        mcp["payload"]["answer"] = "수량 확인 단계의 맥락이 없습니다. 스펙부터 다시 알려주세요."
        return {"mcp": mcp, "last_agent": "cart_quantity_node"}

    qty = parse_int_first(q, item.get("quantity", 1))
    if qty <= 0:
        mcp["payload"]["answer"] = "수량은 1 이상의 정수로 입력해 주세요. (예: 3)"
        return {"mcp": mcp, "last_agent": "cart_quantity_node"}

    kind = ctx.get("kind")
    total_usd = 0.0
    discount_note = ""

    if kind == "consumption":
        usd_per_hour = float(ctx.get("usd_per_hour", 0.0))
        hours = int(ctx.get("hours", 0))
        total_usd = usd_per_hour * hours * qty
        # ★ 종량제 10% 할인 적용
        if CONSUMPTION_DISCOUNT_RATE > 0:
            total_usd = total_usd * (1.0 - CONSUMPTION_DISCOUNT_RATE)
            discount_note = f"(종량제 {int(CONSUMPTION_DISCOUNT_RATE*100)}% 할인 적용)"

    elif kind == "reservation":
        unit_usd = float(ctx.get("usd", 0.0))
        uom = (ctx.get("uom") or "").lower()
        months = int(ctx.get("months", 0))
        hours_per_month = int(ctx.get("hours_per_month", MONTHLY_HOURS))
        # compute_reservation_from_ctx는 qty 포함 총액
        total_usd = compute_reservation_from_ctx(ctx, qty)

    elif kind == "disk":
        mode = ctx.get("mode", "month")
        usd_per_unit = float(ctx.get("usd_per_unit", 0.0))
        months = int(ctx.get("months", 0))
        gb = int(ctx.get("storage_gb", 0))
        if mode == "gb_month":
            total_usd = usd_per_unit * gb * months * qty
        else:
            total_usd = usd_per_unit * months * qty

    elif kind == "aoai":
        usd_per_1k_in = float(ctx.get("usd_per_1k_in", 0.0))
        usd_per_1k_out = float(ctx.get("usd_per_1k_out", 0.0))
        ib = int(ctx.get("ib", 0))
        ob = int(ctx.get("ob", 0))
        usd_input = (ib / 1000.0) * usd_per_1k_in
        usd_output = (ob / 1000.0) * usd_per_1k_out
        total_usd = (usd_input + usd_output) * qty

    else:
        mcp["payload"]["answer"] = "알 수 없는 견적 유형입니다. 처음부터 다시 진행해 주세요."
        mcp["payload"]["awaiting"] = None
        return {"mcp": mcp, "last_agent": "cart_quantity_node"}

    total_krw = to_krw(total_usd)

    # 추천 아이템에 최종 수량/가격 반영 + 단가 메타 저장
    mcp["payload"]["recommended_item"]["quantity"] = qty
    mcp["payload"]["recommended_item"]["price"] = fmt_money_krw(total_krw)
    mcp["payload"]["recommended_item"]["total_usd"] = float(total_usd)
    mcp["payload"]["recommended_item"]["total_krw"] = int(total_krw)

    # 단가 라벨/수치
    if kind == "consumption":
        mcp["payload"]["recommended_item"]["unit_usd"] = float(ctx.get("usd_per_hour", 0.0))
        mcp["payload"]["recommended_item"]["unit_label"] = "per hour"
        # 할인 정보 메모
        mcp["payload"]["recommended_item"]["discount_applied"] = {
            "type": "consumption_10pct",
            "rate": CONSUMPTION_DISCOUNT_RATE,
            "note": "종량제 10% 할인 적용"
        }
    elif kind == "reservation":
        mcp["payload"]["recommended_item"]["unit_usd"] = float(ctx.get("usd", 0.0))
        mcp["payload"]["recommended_item"]["unit_label"] = ctx.get("uom", "per unit")
    elif kind == "disk":
        mcp["payload"]["recommended_item"]["unit_usd"] = float(ctx.get("usd_per_unit", 0.0))
        mcp["payload"]["recommended_item"]["unit_label"] = "per month" if ctx.get("mode") == "month" else "per GB·month"
    elif kind == "aoai":
        mcp["payload"]["recommended_item"]["unit_usd_in"]  = float(ctx.get("usd_per_1k_in", 0.0))
        mcp["payload"]["recommended_item"]["unit_usd_out"] = float(ctx.get("usd_per_1k_out", 0.0))
        mcp["payload"]["recommended_item"]["unit_label"]   = "per 1K tokens"

    # KRW 단가 파생치
    if "unit_usd" in mcp["payload"]["recommended_item"]:
        mcp["payload"]["recommended_item"]["unit_krw"] = int(to_krw(mcp["payload"]["recommended_item"]["unit_usd"]))
    if "unit_usd_in" in mcp["payload"]["recommended_item"]:
        mcp["payload"]["recommended_item"]["unit_krw_in"]  = int(to_krw(mcp["payload"]["recommended_item"]["unit_usd_in"]))
    if "unit_usd_out" in mcp["payload"]["recommended_item"]:
        mcp["payload"]["recommended_item"]["unit_krw_out"] = int(to_krw(mcp["payload"]["recommended_item"]["unit_usd_out"]))

    # 표/CSV용 스펙 라벨 보강
    item = mcp["payload"]["recommended_item"] or {}
    pending_safe = mcp["payload"].get("pending") or {}
    spec_label = (
        pending_safe.get("sku_name")
        or item.get("spec")
        or item.get("option")
        or pending_safe.get("product_name")
        or item.get("service")
        or ""
    )
    item["spec"] = spec_label
    item["quantity"] = int(qty)

    # 1개 기준 총액(표시/증감 계산용)
    per_unit_usd = (float(total_usd) / max(1, int(qty))) if total_usd else 0.0
    item["unit_usd"] = per_unit_usd
    item["unit_krw"] = int(to_krw(per_unit_usd))
    mcp["payload"]["recommended_item"] = item

    # 다음: 장바구니 추가 여부 확인 (할인 메모 포함)
    mcp["payload"]["awaiting"] = "cart_confirm"
    discount_line = f" {discount_note}" if discount_note else ""
    mcp["payload"]["answer"] = (
        f"수량 {qty}개로 반영한 최종 견적은 {fmt_usd(total_usd)} ≈ {fmt_money_krw(total_krw)} 입니다.{discount_line}\n\n"
        "장바구니에 추가할까요? (네/아니오)"
    )
    return {"mcp": mcp, "last_agent": "cart_quantity_node"}


#네/아니오 해석해 실제 추가/취소 하는 노드
def confirm_cart_node(state: GraphState) -> GraphState:
    mcp = state["mcp"]
    q = mcp["payload"]["question"]
    session_id = mcp["payload"]["metadata"]["session_id"]
    awaiting = mcp["payload"].get("awaiting")
    item = mcp["payload"].get("recommended_item")

    if awaiting != "cart_confirm" or not item:
        # 맥락이 없으면 일반 응답으로 회귀
        mcp["payload"]["answer"] = "추가할 항목이 없습니다. 스펙을 먼저 알려주세요."
        return {"mcp": mcp, "last_agent": "confirm_cart_node"}

    if user_says_no(q):
        # 취소
        mcp["payload"]["awaiting"] = None
        mcp["payload"]["answer"] = "장바구니에 추가하지 않았습니다."
        # 필요하면 recommended_item도 정리
        # mcp["payload"]["recommended_item"] = None
        return {"mcp": mcp, "last_agent": "confirm_cart_node"}

    if user_says_yes(q):
        # 안전한 복사 후 장바구니에 즉시 반영
        item_copy = copy.deepcopy(item)

        # 혹시 숫자 필드가 없으면 복구(보수적으로)
        if "total_usd" not in item_copy or "total_krw" not in item_copy:
            qty = int(item_copy.get("quantity", 1))
            unit_usd = float(item_copy.get("unit_usd", 0.0))
            item_copy["total_usd"] = round(unit_usd * qty, 6)
            item_copy["total_krw"] = int(to_krw(item_copy["total_usd"]))

        shopping_carts.setdefault(session_id, []).append(item_copy)

        mcp["payload"]["awaiting"] = None
        mcp["payload"]["shopping_cart"] = shopping_carts[session_id]
        mcp["payload"]["answer"] = (
            f"{item_copy.get('service','항목')}이(가) 장바구니에 추가되었습니다. "
            f"현재 장바구니 {len(shopping_carts[session_id])}개."
        )
        return {"mcp": mcp, "last_agent": "confirm_cart_node"}

    """
    if user_says_yes(q):
        # 실제 추가
        _ensure_session_cart(session_id)
        cart_row = _build_cart_row(
            session_id=session_id,
            item=item,                 # recommended_item
            ctx=mcp["payload"].get("cart_ctx", {})
        )
        shopping_carts[session_id].append(cart_row)

        mcp["payload"]["awaiting"] = None
        mcp["payload"]["shopping_cart"] = shopping_carts[session_id]
        totals = _cart_totals(shopping_carts[session_id])
        mcp["payload"]["cart_totals"] = totals            
        mcp["payload"]["answer"] = (
            f"{cart_row['service']}이(가) 장바구니에 추가되었습니다. "
            f"현재 장바구니 {len(shopping_carts[session_id])}개. "
            f"(합계: ${totals['total_usd']:,.4f} ≈ {fmt_money_krw(totals['total_krw'])})"
        )            
        return {"mcp": mcp, "last_agent": "confirm_cart_node"}            
    """
    # 명확히 못 알아들었으면 다시 물음
    mcp["payload"]["answer"] = "장바구니에 추가할까요? (네/아니오)"
    return {"mcp": mcp, "last_agent": "confirm_cart_node"}

def compute_reservation_from_ctx(ctx: dict, qty: int) -> float:
    """
    cart_ctx로 저장해 둔 예약제 정보를 바탕으로 총액(USD)을 재계산.
    ctx = {
      "kind": "reservation",
      "months": 12 or 36,
      "usd": <RetailPrice>,
      "uom": "hour"/"month"/기타(일시금),
    }
    """
    months = int(ctx.get("months", 0))
    unit_usd = float(ctx.get("usd", 0.0))
    uom = (ctx.get("uom") or "").lower()

    if months <= 0 or unit_usd <= 0:
        return 0.0

    if "hour" in uom:
        return unit_usd * MONTHLY_HOURS * months * qty
    if "month" in uom:
        return unit_usd * months * qty
    return unit_usd * qty

def compute_reservation_total(price_entry: dict, months: int, quantity: int) -> tuple[int, float, str]:
    if not price_entry or not price_entry.get("usd"):
        return 0, 0.0, "가격 데이터 없음"
    usd = float(price_entry["usd"] or 0)
    uom = (price_entry.get("uom") or "").lower()
    if "hour" in uom:
        total_usd = usd * MONTHLY_HOURS * months * quantity
        detail = f"{MONTHLY_HOURS}시간×{months}개월×{quantity}대"
    elif "month" in uom:
        total_usd = usd * months * quantity
        detail = f"{months}개월×{quantity}대"
    else:
        total_usd = usd * quantity
        detail = f"{quantity}대"
    return to_krw(total_usd), float(total_usd), detail

#---- 장바구니 표 행 포맷 함수 & 합계 계산 유틸 추가
def _now_kst_str():
    KST = timezone(timedelta(hours=9))
    return datetime.now(KST).strftime("%Y-%m-%d")

def _ensure_session_cart(session_id: str):
    shopping_carts.setdefault(session_id, [])
    cart_meta.setdefault(session_id, {"months": None})

def _calc_line_totals_from_ctx(ctx: dict, qty: int) -> tuple[float, float]:
    """
    ctx(kind)에 따라 USD 총액 계산 -> (total_usd, unit_usd) 반환
    """
    kind = ctx.get("kind")
    if kind == "consumption":
        usd_per_hour = float(ctx.get("usd_per_hour", 0.0))
        hours = int(ctx.get("hours", 0))
        unit_usd = usd_per_hour * hours  # '1개' 기준
        return unit_usd * qty, usd_per_hour
    elif kind == "reservation":
        # month/hour/one-time 모두 compute_reservation_from_ctx로 총액만 계산
        months = int(ctx.get("months", 0))
        # compute_reservation_from_ctx는 qty 포함 총액을 주므로 unit 추정을 위해 qty=1로도 한 번 계산
        total_all = compute_reservation_from_ctx(ctx, qty)
        unit_one = compute_reservation_from_ctx(ctx, 1)
        # 'unit_usd'는 1개 기준 총액(전체 기간)
        return total_all, unit_one
    elif kind == "disk":
        mode = ctx.get("mode", "month")
        usd_per_unit = float(ctx.get("usd_per_unit", 0.0))
        months = int(ctx.get("months", 0))
        gb = int(ctx.get("storage_gb", 0))
        if mode == "gb_month":
            unit_one = usd_per_unit * gb * months
        else:
            unit_one = usd_per_unit * months
        return unit_one * qty, unit_one
    elif kind == "aoai":
        usd_per_1k_in = float(ctx.get("usd_per_1k_in", 0.0))
        usd_per_1k_out = float(ctx.get("usd_per_1k_out", 0.0))
        ib = int(ctx.get("ib", 0))
        ob = int(ctx.get("ob", 0))
        unit_one = (ib/1000.0) * usd_per_1k_in + (ob/1000.0) * usd_per_1k_out
        return unit_one * qty, unit_one
    return 0.0, 0.0

def _build_cart_row(session_id: str, item: dict, ctx: dict) -> dict:
    """
    기존 recommended_item + cart_ctx로 테이블 1행 구조화
    표 요구사항:
      서비스명(VM/DB/Storage/AOAI/Databricks), 스펙(option), 수량, 가격(USD), 가격(KRW)
    """
    qty = int(item.get("quantity", 1))
    total_usd, unit_or_ref = _calc_line_totals_from_ctx(ctx, qty)
    # 표시는 총액(USD/KRW). 수량 조절 시 총액 재계산됨.
    row = {
        "service": item.get("service") or "Service",
        "spec": item.get("option") or "",             # = 스펙
        "quantity": qty,
        "price_usd": round(total_usd, 6),             # 총액(USD)
        "price_krw": to_krw(total_usd),               # 총액(KRW)
        # 프론트에서 필요 시 참조
        "_unit_ref": unit_or_ref,                     # 예약제/시간제에서 '단가' 또는 '1개 기준 총액'
        "_ctx": ctx,                                  # 재계산용(수정/업데이트 시 활용)
    }
    return row

def _recompute_row(row: dict, new_qty: int) -> dict:
    ctx = row.get("_ctx", {})
    total_usd, _unit = _calc_line_totals_from_ctx(ctx, new_qty)
    row["quantity"] = int(new_qty)
    row["price_usd"] = round(total_usd, 6)
    row["price_krw"] = to_krw(total_usd)
    return row

def _cart_totals(cart: List[dict]) -> dict:
    usd = sum(float(x.get("price_usd", 0.0)) for x in cart)
    krw = to_krw(usd)
    return {"total_usd": round(usd, 6), "total_krw": krw}

# ---- 3-2. 장바구니 추가 ----
def add_to_cart_node(state: GraphState) -> GraphState:
    mcp = state["mcp"]
    session_id = mcp["payload"]["metadata"]["session_id"]
    recommended_item = mcp["payload"].get("recommended_item")

    if not recommended_item:
        mcp["payload"]["answer"] = "추가할 항목이 없습니다."
        return {"mcp": mcp, "last_agent": "add_to_cart_node"}

    shopping_carts.setdefault(session_id, []).append(recommended_item)
    answer = f"{recommended_item['service']}이(가) 장바구니에 추가되었습니다. 현재 장바구니 {len(shopping_carts[session_id])}개."
    mcp["payload"]["answer"] = answer
    mcp["payload"]["shopping_cart"] = shopping_carts[session_id]
    return {"mcp": mcp, "last_agent": "add_to_cart_node"}

# ---- 4. 장바구니 보기 ----
def view_cart_node(state: GraphState) -> GraphState:
    mcp = state["mcp"]
    session_id = mcp["payload"]["metadata"]["session_id"]
    cart = shopping_carts.get(session_id, [])
    if not cart:
        answer = "장바구니가 비어있습니다."
    else:
        lines = [
            f"- {i.get('service','Service')} ({i.get('spec') or i.get('option') or ''}, {i.get('quantity',1)}개) - {i.get('price') or fmt_money_krw(i.get('total_krw',0))}"
            for i in cart
        ]
        answer = "현재 장바구니:\n" + "\n".join(lines)
    mcp["payload"]["answer"] = answer
    return {"mcp": mcp, "last_agent": "view_cart_node"}


# ---- LLM Intent 분류 ---
def classify_intent_with_llm(text: str) -> str:
    system_prompt = """
You classify into one of:
- azure_inquiry_agent
- site_inquiry_agent
- spec_matching_agent
- view_cart_agent
- yes_no_agent

Examples:
"Azure VM이 뭐야?" -> azure_inquiry_agent
"사이트 사용법" -> site_inquiry_agent
"서버 견적" -> spec_matching_agent
"장바구니 보여줘" -> view_cart_agent
"네" -> yes_no_agent

Return only the label.
"""
    try:
        msgs = [SystemMessage(content=system_prompt), HumanMessage(content=text)]
        resp = azure_llm.invoke(msgs).content.strip().lower()
        if "azure_inquiry_agent" in resp:
            return "azure_inquiry_agent"
        if "site_inquiry_agent" in resp:
            return "site_inquiry_agent"
        if "spec_matching_agent" in resp:
            return "spec_matching_agent"
        if "view_cart_agent" in resp:
            return "view_cart_agent"
        if "yes_no_agent" in resp:
            return "yes_no_agent"
        return "azure_inquiry_agent"
    except Exception as e:
        logging.warning(f"LLM 라우팅 실패: {e}")
        return "azure_inquiry_agent"

# ---- Graph ----
builder = StateGraph(GraphState)
builder.add_node("azure_inquiry_node", azure_inquiry_node)
builder.add_node("site_inquiry_node", site_inquiry_node)
builder.add_node("spec_matching_node", spec_matching_node)
builder.add_node("add_to_cart_node", add_to_cart_node)
builder.add_node("view_cart_node", view_cart_node)
builder.add_node("plan_or_hours_node", plan_or_hours_node)
builder.add_node("confirm_cart_node", confirm_cart_node)
builder.add_node("cart_quantity_node", cart_quantity_node)

def route_to_agent(state: GraphState) -> str:
    q = state["mcp"]["payload"]["question"]
    awaiting = state["mcp"]["payload"].get("awaiting")

    # 장바구니 분기 우선
    if awaiting == "cart_quantity":
        return "cart_quantity_node"
    if awaiting == "cart_confirm":
        return "confirm_cart_node"

    # ★ 견적 중간단계(시간/개월/토큰/플랜)는 전부 plan_or_hours_node로 보낸다
    # ★ 견적 중간단계(시간/개월/토큰/플랜)는 전부 plan_or_hours_node로 보낸다
    if awaiting in {"hours", "months", "disk_months", "tokens", "plan", "dbx_plan"}:
        return "plan_or_hours_node"

    # 그 외는 일반 라우팅
    return classify_intent_with_llm(q)

def route_after_spec_matching(state: GraphState) -> str:
    # 현재 턴의 질문만 보고 yes/no 판정하기 어렵지만, 단일 턴 흐름으로 간소화
    user_response = state["mcp"]["payload"]["question"].lower()
    return "yes" if ("네" in user_response or "추가" in user_response) else "no"

builder.add_conditional_edges(
    START,
    route_to_agent,
    {
        "azure_inquiry_agent": "azure_inquiry_node",
        "site_inquiry_agent": "site_inquiry_node",
        "spec_matching_agent": "spec_matching_node",
        "view_cart_agent": "view_cart_node",
        "yes_no_agent": "confirm_cart_node",
        "plan_or_hours_node": "plan_or_hours_node",
        "confirm_cart_node": "confirm_cart_node",
        "cart_quantity_node": "cart_quantity_node",
    }
)

# spec_matching_node 이후에는 END (이제 plan 선택은 다음 턴에 새 라우터가 처리)
builder.add_edge("spec_matching_node", END)
builder.add_edge("azure_inquiry_node", END)
builder.add_edge("site_inquiry_node", END)
builder.add_edge("add_to_cart_node", END)
builder.add_edge("view_cart_node", END)

main_workflow = builder.compile()

# ---- Schemas & Endpoints ----
class QARequest(BaseModel):
    question: str
    session_id: str

class EmbedRequest(BaseModel):
    text: List[str]

class CartUpdateReq(BaseModel):
    session_id: str
    index: int
    quantity: int

class CartDeleteReq(BaseModel):
    session_id: str
    index: int

class CartMetaReq(BaseModel):
    session_id: str
    months: int | None = None   # 사업기간 (개월). None이면 미설정


@app.get("/cart")
def get_cart(session_id: str):
    _ensure_session_cart(session_id)
    cart = shopping_carts.get(session_id, [])
    totals = _cart_totals(cart)
    # 프론트 표시는 이 필드만 쓰면 됨
    rows = [
        {
            "service": r.get("service", "Service"),
            "spec": r.get("spec") or r.get("option") or "",  # ✅ 폴백
            "quantity": r.get("quantity", 1),
            "price_usd": r.get("price_usd") or float(r.get("total_usd", 0.0)),  # ✅ 숫자 폴백
            "price_krw": r.get("price_krw") or int(r.get("total_krw", 0)),      # ✅ 숫자 폴백
        } for r in cart
    ]
    return {
        "date": _now_kst_str(),
        "krw_rate": KRW_RATE,
        "project_months": cart_meta.get(session_id, {}).get("months"),
        "rows": rows,
        "totals": totals
    }

@app.post("/cart/update")
def update_cart_quantity(req: CartUpdateReq):
    _ensure_session_cart(req.session_id)
    cart = shopping_carts.get(req.session_id, [])
    if req.index < 0 or req.index >= len(cart):
        raise HTTPException(status_code=400, detail="invalid index")
    if req.quantity <= 0:
        raise HTTPException(status_code=400, detail="quantity must be >= 1")
    cart[req.index] = _recompute_row(cart[req.index], req.quantity)
    totals = _cart_totals(cart)
    return {"rows": cart, "totals": totals}

@app.post("/cart/delete")
def delete_cart_item(req: CartDeleteReq):
    _ensure_session_cart(req.session_id)
    cart = shopping_carts.get(req.session_id, [])
    if req.index < 0 or req.index >= len(cart):
        raise HTTPException(status_code=400, detail="invalid index")
    cart.pop(req.index)
    totals = _cart_totals(cart)
    return {"rows": cart, "totals": totals}

@app.post("/cart/meta")
def set_project_months(req: CartMetaReq):
    _ensure_session_cart(req.session_id)
    cart_meta[req.session_id]["months"] = req.months
    return {
        "date": _now_kst_str(),
        "krw_rate": KRW_RATE,
        "project_months": req.months
    }

@app.get("/cart/export.csv")
def export_cart_csv(session_id: str):
    _ensure_session_cart(session_id)
    cart = shopping_carts.get(session_id, [])
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["날짜", _now_kst_str()])
    writer.writerow(["환율(USD→KRW)", int(KRW_RATE)])
    writer.writerow(["사업기간(개월)", cart_meta.get(session_id, {}).get("months") or ""])
    writer.writerow([])
    writer.writerow(["서비스명","스펙","수량","가격(USD)","가격(KRW)"])
    for r in cart:
        writer.writerow([
            r["service"], r["spec"], r["quantity"],
            f"{r['price_usd']:.6f}", r["price_krw"]
        ])
    totals = _cart_totals(cart)
    writer.writerow([])
    writer.writerow(["총합","","", f"{totals['total_usd']:.6f}", totals["total_krw"]])
    csv_bytes = io.BytesIO(output.getvalue().encode("utf-8-sig"))
    headers = {
        "Content-Disposition": "attachment; filename=azure_estimate_cart.csv"
    }
    return StreamingResponse(csv_bytes, media_type="text/csv", headers=headers)


@app.get("/")
def health():
    return {"message": "MCP 기반 LangGraph QA 서버입니다."}

@app.post("/answer")
def answer_question(req: QARequest):
    session_id = req.session_id

    cancel_words = ["취소", "초기화", "reset", "리셋", "다시 시작"]
    if any(w in (req.question or "").lower() for w in cancel_words):
        SESSION_PAYLOADS.pop(session_id, None)
        pending_by_session_id.pop(session_id, None)
        return {
            "question": req.question,
            "metadata": {"session_id": session_id},
            "answer": "세션 상태를 초기화했어요. 무엇부터 견적낼까요?\n예) `storage ssd 500GB`, `mysql 100시간`, `Databricks 200시간`"
        }

    # 1) 이전 payload 복원
    base_payload = {
        "question": req.question,
        "metadata": {"session_id": session_id}
    }
    prev_payload = SESSION_PAYLOADS.get(session_id) or {}
    for k, v in prev_payload.items():
        if k not in ("question", "metadata"):
            base_payload[k] = v

    # >>> [추가] pending/awaiting이 비었으면 보조 저장소에서 복구
    if not base_payload.get("pending"):
        maybe = pending_by_session_id.get(session_id)
        if maybe:
            base_payload["pending"] = maybe
            base_payload.setdefault("awaiting", maybe.get("awaiting"))
    mcp = {
        "source": "user",
        "destination": "",
        "intent": "",
        "payload": base_payload
    }

    try:
        state = main_workflow.invoke({"mcp": mcp})
        payload = state["mcp"]["payload"]

        # ▶▶ 추가: 환율, 오늘 날짜 포함
        payload.setdefault("fx_rate", KRW_RATE)
        payload.setdefault("today", datetime.now().strftime("%Y-%m-%d"))

        SESSION_PAYLOADS[session_id] = payload
        return payload
    except Exception as e:
        logging.exception("answer 처리 중 오류")
        raise HTTPException(status_code=500, detail=str(e))

# 선택: /embed 구현 (Streamlit '데이터 저장' 버튼용)
from langchain_community.vectorstores import Chroma as _ChromaRaw
from langchain.docstore.document import Document as LCDocument

@app.post("/embed")
def embed_texts(req: EmbedRequest):
    try:
        texts = [t for t in req.text if isinstance(t, str) and t.strip()]
        if not texts:
            return JSONResponse({"status":"ok","message":"no texts"}, status_code=200)
        docs = [LCDocument(page_content=t) for t in texts]
        _ChromaRaw.from_documents(
            documents=docs,
            embedding=embedding_model,
            persist_directory="./chromaDB"
        )
        return {"status":"ok","count": len(texts)}
    except Exception as e:
        logging.exception("/embed 오류")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset")
def reset_endpoint(req: QARequest):
    reset_session(req.session_id)
    return {"status": "ok", "message": "세션 초기화 완료"}

