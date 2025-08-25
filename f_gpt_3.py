import os
import uuid
import json
import requests
import datetime as dt
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

# ================== 기본 설정 ==================
API_SERVER = os.getenv("API_SERVER", "http://localhost:8001")
DEFAULT_USD_KRW = float(os.getenv("USD_KRW", "1350"))

st.set_page_config(
    page_title="Azure딱깔센 챗봇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 사이드바 넓히기 + 본문 최소 폭 좁게
st.markdown("""
<style>
/* 사이드바 너비 확장 */
[data-testid="stSidebar"] { width: 520px; min-width: 520px; }
/* 메인 컨테이너 여백 조금 */
.block-container { padding-top: 1rem; padding-bottom: 2rem; }
/* 표 버튼 영역 정렬 */
.btn-cell { display:flex; gap:6px; align-items:center; }
.small-btn button { padding:2px 6px; }
</style>
""", unsafe_allow_html=True)

# ================== 세션 상태 ==================
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# 서버가 내려준 장바구니(단일 소스)
if "shopping_cart" not in st.session_state:
    st.session_state.shopping_cart = []

# 사이드바에서 사용하는 상태
if "project_months" not in st.session_state:
    st.session_state.project_months = 3

if "usd_krw" not in st.session_state:
    st.session_state.usd_krw = DEFAULT_USD_KRW

# ================== 공통 함수 ==================
def post_answer(question: str) -> Dict[str, Any]:
    payload = {"question": question, "session_id": st.session_state.session_id}
    res = requests.post(f"{API_SERVER}/answer", json=payload, timeout=60)
    res.raise_for_status()
    return res.json()

def cart_to_dataframe(cart: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    서버 패치 이후 아이템은 다음 키들을 가짐:
    - service (VM/DB/Storage/AOAI/Databricks 등)
    - spec (SKU or 설명)
    - quantity (int)
    - total_usd (float), total_krw (int)
    - unit_usd/unit_krw/unit_label (있으면 사용)
    - AOAI일 때 unit_usd_in/out 도 올 수 있음
    """
    rows = []
    for i, it in enumerate(cart):
        service = it.get("service", "N/A")
        spec    = it.get("spec") or it.get("option","")
        qty     = int(it.get("quantity", 1))

        # 총액 (서버가 보장)
        total_usd = float(it.get("total_usd", it.get("price_usd", 0.0)))
        total_krw = int(it.get("total_krw",  it.get("price_krw", 0)))

        rows.append({
            "idx": i,
            "서비스": service,
            "스펙": spec,
            "수량": qty,
            "가격(USD)": round(total_usd, 4),
            "가격(KRW)": total_krw
        })
    df = pd.DataFrame(rows, columns=["idx", "서비스", "스펙", "수량", "가격(USD)", "가격(KRW)"])
    return df

def compute_totals(cart: List[Dict[str, Any]]) -> Dict[str, float]:
    usd = 0.0
    krw = 0
    for it in cart:
        usd += float(it.get("total_usd", it.get("price_usd", 0.0)))
        krw += int(it.get("total_krw",  it.get("price_krw", 0)))
    return {"usd": usd, "krw": krw}

def download_csv(cart_df: pd.DataFrame) -> bytes:
    return cart_df.drop(columns=["idx"]).to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")

# ================== 사이드바 UI ==================
with st.sidebar:
    # 상단 헤더
    today = dt.datetime.now().strftime("%Y-%m-%d")
    st.markdown(f"### 🧾 견적 정보")
    st.write(f"- 오늘 날짜: **{today}**")
    st.write(f"- 적용 환율: **1 USD = {int(st.session_state.usd_krw):,} KRW**")
    st.session_state.project_months = st.number_input("사업기간 (개월)", min_value=1, max_value=60, value=st.session_state.project_months)

    st.markdown("---")
    st.markdown("### 🛒 장바구니")

    # 표 렌더
    df_cart = cart_to_dataframe(st.session_state.shopping_cart)
    if df_cart.empty:
        st.info("장바구니가 비어있습니다.")
    else:
        # 행별 조작 버튼 + 수량 조절
        new_cart = st.session_state.shopping_cart.copy()
        for _, row in df_cart.iterrows():
            idx = int(row["idx"])
            col1, col2 = st.columns([1, 2.5])
            with col1:
                st.caption(f"#{idx+1}")
                # 삭제 버튼
                if st.button("삭제", key=f"del_{idx}", help="항목 삭제"):
                    # 서버 세션의 장바구니도 업데이트 하려면 전용 API가 필요.
                    # 현재는 UI상에서만 제거하고 CSV/합계에 반영.
                    new_cart.pop(idx)
                    st.session_state.shopping_cart = new_cart
                    st.rerun()
            with col2:
                st.markdown(f"**{row['서비스']}**")
                st.caption(row["스펙"])
                c1, c2, c3, c4 = st.columns([1,1,2,2])
                with c1:
                    if st.button("－", key=f"minus_{idx}"):
                        q = max(1, int(new_cart[idx].get("quantity", 1)) - 1)
                        u = float(new_cart[idx].get("unit_usd", 0.0))
                        new_cart[idx]["quantity"] = q
                        new_cart[idx]["total_usd"] = round(u * q, 6)
                        new_cart[idx]["total_krw"] = int(new_cart[idx]["total_usd"] * st.session_state.usd_krw)
                        st.session_state.shopping_cart = new_cart
                        st.rerun()
                with c2:
                    if st.button("＋", key=f"plus_{idx}"):
                        q = int(new_cart[idx].get("quantity", 1)) + 1
                        u = float(new_cart[idx].get("unit_usd", 0.0))
                        new_cart[idx]["quantity"] = q
                        new_cart[idx]["total_usd"] = round(u * q, 6)
                        new_cart[idx]["total_krw"] = int(new_cart[idx]["total_usd"] * st.session_state.usd_krw)
                        st.session_state.shopping_cart = new_cart
                        st.rerun()
                with c3:
                    st.metric("가격(USD)", f"{row['가격(USD)']:,}")
                with c4:
                    st.metric("가격(KRW)", f"{int(row['가격(KRW)']):,}원")

        # 총합 & CSV 내보내기
        st.markdown("---")
        totals = compute_totals(st.session_state.shopping_cart)
        st.subheader("합계")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("총액(USD)", f"{totals['usd']:.4f}")
        with c2:
            st.metric("총액(KRW)", f"{int(totals['krw']):,}원")

        csv_bytes = download_csv(cart_to_dataframe(st.session_state.shopping_cart))
        st.download_button(
            label="CSV로 내보내기",
            data=csv_bytes,
            file_name=f"azure_quote_{today}.csv",
            mime="text/csv"
        )

# ================== 메인: 챗봇만 ==================
st.title("LangGraph 기반 MCP QA 시스템")

# 기존 대화 렌더
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("질문을 입력하세요. (예: '4코어 VM 견적 내줘' / '네')")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    try:
        result = post_answer(user_input)
        # 응답 표시
        answer = result.get("answer", "")
        with st.chat_message("assistant"):
            st.markdown(f"**답변:** {answer}")

        # ❗ 서버가 같은 턴에 shopping_cart를 내려주면 즉시 반영
        if "shopping_cart" in result:
            st.session_state.shopping_cart = result["shopping_cart"]

        # 대화 기록 저장
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"**답변:** {answer}"
        })

        # "장바구니에 추가되었습니다" 같은 문구가 오면 즉시 리프레시해 사이드바에 반영
        if "추가되었습니다" in answer or ("shopping_cart" in result and result["shopping_cart"]):
            st.rerun()

    except requests.exceptions.RequestException as e:
        st.error(f"FastAPI 응답 실패: {e}")
