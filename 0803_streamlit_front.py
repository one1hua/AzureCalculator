import streamlit as st
import requests
import uuid

# 페이지 설정
st.set_page_config(page_title="애저딱칼센", layout="wide")
st.sidebar.header("Side bar")
mode = st.sidebar.radio("기능", ["장바구니", "질문하기"])

API_SERVER = "http://40.82.143.146/api"  # 절대경로로 명확히 설정

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# CSS 적용
st.markdown("""
    <style>
        body, .stApp { background-color: black !important; color: white !important; }
        header[data-testid="stHeader"] { background-color: #00B8CC !important; height: 70px; }
        [data-testid="stSidebar"] { background-color: #EBFCFF !important; color: black !important; }
        [data-testid="stSidebar"] * { color: black !important;
  }
        .stChatInputContainer { background-color: #00B8CC !important; padding: 20px !important; border-radius: 20px; }
        [data-testid="stChatInput"] textarea { background-color: white !important; color: black !important; }
        [data-testid="stChatInputSubmitButton"] { background-color: white !important; color: #00B8CC !important; }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    .st-emotion-cache-janbn0 {
            flex-direction: row-reverse;
            text-align: right;
        }
    </style>
""", unsafe_allow_html=True)

# 장바구니 모드
if mode == "장바구니":
    st.header("🛍️  Azure 상품 장바구니 담기")
    st.write('여기는 성은과장님 개발 예정인 부분이라 아직 내용은 없습니다~')

# 질문하기 모드
elif mode == "질문하기":
    st.header("📝 Azure 기초 견적 계산 질문")

    # 이전 대화 출력
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

        with st.spinner("답변 생성중..."):
            try:
                res = requests.post(f"{API_SERVER}/answer", json=payload, timeout=60)
                res.raise_for_status()
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

            except requests.exceptions.RequestException as e:
                st.error(f"FastAPI 응답 실패: {e}")



# 이전 대화 구분선
if len(st.session_state.get("messages", [])) > 1:
    st.markdown("""
        <div style="display: flex; align-items: center; text-align: center; margin: 20px 0;">
            <hr style="flex-grow: 1; border: 0.5px solid gray; margin: 0 10px;">
            <span style="white-space: nowrap; color: gray; font-size: 14px;">이전 대화</span>
            <hr style="flex-grow: 1; border: 0.5px solid gray; margin: 0 10px;">
        </div>
    """, unsafe_allow_html=True)

