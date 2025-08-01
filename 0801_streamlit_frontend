import streamlit as st
import pandas as pd
import requests
import uuid

# 페이지 설정
st.set_page_config(page_title="애저딱칼센", layout="wide")
st.sidebar.header("Side bar")
mode = st.sidebar.radio("기능", ["장바구니", "질문하기"])

API_SERVER = "http://localhost:8000"

# 🔥 장바구니 모드
if mode == "장바구니":
    st.header("🛍️ Azure 상품 장바구니 담기")
    st.write('여기는 성은과장님 개발 예정인 부분이라 아직 내용은 없습니다~')

# 🔥 질문하기 모드
elif mode == "질문하기":
    st.header("📝 Azure 기초 견적 계산 질문")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("질문을 입력하세요:")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        payload = {
            "question": user_input,
            "session_id": st.session_state.session_id
        }

        res = requests.post(f"{API_SERVER}/answer", json=payload)

        if res.status_code == 200:
            result = res.json()
            answer = result.get("answer")
            references = result.get("references", [])

            response = f"**답변:** {answer}"
            st.session_state.messages.append({"role": "assistant", "content": response})

            with st.chat_message("assistant"):
                st.markdown(response)
                if references:
                    with st.expander("참고 문서 보기"):
                        for i, ref in enumerate(references, 1):
                            st.markdown(f"{i}. {ref}")
        else:
            st.error("FastAPI 응답 실패")

# 🔥 질문하기 페이지 꾸미기 - CSS 삽입
st.markdown("""
    <style>
        /* 전체 배경과 텍스트 색상 */
        body, .stApp {
            background-color: black !important;
            color: white !important;
        }

        /* 상단 네비게이션 바 */
        header[data-testid="stHeader"] {
            background-color: #00B8CC !important;
            height: 70px;
        }

        /* 툴바 */
        [data-testid="stToolbar"] {
            background-color: #00B8CC !important;
            color: black !important;
        }

        /* 사이드바 스타일 */
        [data-testid="stSidebar"] {
            background-color: #EBFCFF !important;
            color: white !important;
            border-right: 1px solid white !important;
        }

        /* 사이드바 내부 글씨 색상 */
        [data-testid="stSidebar"] * {
            color: black !important;
        }

        /* 텍스트 커서 색 */
        div[data-baseweb="textarea"] textarea {
            caret-color: #00B8CC !important;
        }

        /* 플레이스홀더 색 */
        div[data-baseweb="textarea"] textarea::placeholder {
            color: #95B0B5 !important;
        }

        /* 하단 입력창 배경 (민트색 바탕) */
        .stChatInputContainer {
            background-color: #00B8CC !important;
            padding: 20px !important;
            border-top-left-radius: 20px;
            border-top-right-radius: 20px;
        }

		footer, .st-emotion-cache-1dp5vir {
   			 background-color: #00B8CC !important;
		}

        /* 채팅 입력창 내부 스타일 */
        [data-testid="stChatInput"] textarea {
            background-color: white !important;
            color: black !important;
            border: 2px solid white !important;
            border-radius: 10px !important;
            padding: 10px 12px !important;
            height: auto !important;
            min-height: 40px !important;
            width: 100% !important;
            box-sizing: border-box !important;
        }

        /* 입력 버튼 스타일 */
        [data-testid="stChatInputSubmitButton"] {
            background-color: white !important;
            color: #00B8CC !important;
            border-radius: 5px !important;
            padding: 5px 10px !important;
            position: absolute !important;
            right: 2px !important;
            bottom: 5px !important;
            transition: transform 0.2s ease-in-out;
        }

        /* 입력 버튼 hover 효과 */
        [data-testid="stChatInputSubmitButton"]:hover {
            background-color: #daebed !important;
            color: #00B8CC !important;
            transform: scale(1.1) !important;
        }

        /* 사용자 채팅 말풍선 정렬 */
        .st-emotion-cache-janbn0 {
            flex-direction: row-reverse;
            text-align: right;
        }
    </style>
""", unsafe_allow_html=True)

# 🔥 이전 대화 라벨
if len(st.session_state.get("messages", [])) > 1:
    st.markdown("""
        <div style="display: flex; align-items: center; text-align: center; margin: 20px 0;">
            <hr style="flex-grow: 1; border: 0.5px solid gray; margin: 0 10px;">
            <span style="white-space: nowrap; color: gray; font-size: 14px;">이전 대화</span>
            <hr style="flex-grow: 1; border: 0.5px solid gray; margin: 0 10px;">
        </div>
    """, unsafe_allow_html=True)
