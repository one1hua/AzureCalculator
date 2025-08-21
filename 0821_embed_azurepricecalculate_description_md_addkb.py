import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document

# === Azure OpenAI 설정 (하드코딩 버전: .env 없이 사용) ===
AZURE_API_KEY = "8D0TqUJQKj7SfHCMcRSAs5cJBSbvvQIcsuZ6QvHFKWsVsP9JyhooJQQJ99BGACNns7RXJ3w3AAABACOG0jZk"
AZURE_ENDPOINT_BASE = "https://azure-openai-price01.openai.azure.com"  # 베이스 URL만!
EMBED_DEPLOYMENT = "text-embedding-3-small"
API_VERSION = "2024-12-01-preview"

embedding_model = AzureOpenAIEmbeddings(
    azure_endpoint=AZURE_ENDPOINT_BASE,
    azure_deployment=EMBED_DEPLOYMENT,
    api_key=AZURE_API_KEY,
    openai_api_version=API_VERSION,
)

# === 입력/출력 경로 & KB 이름 ===
MD_PATH = "./azurepricecalculate_description.md"
PERSIST_DIR = "./chromaDB"
KB_NAME = "kb_md"  # ← MD 코퍼스용 KB 이름

# === Markdown 로드 ===
with open(MD_PATH, "r", encoding="utf-8") as f:
    text = f.read()

# === 청크 분할 (필요시 조정) ===
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = splitter.split_text(text)
print(f"[INFO] 총 {len(chunks)}개의 청크로 분할")

# === Document로 변환 (메타데이터에 KB/소스/청크인덱스 추가) ===
docs = []
src_abs = os.path.abspath(MD_PATH)
for i, content in enumerate(chunks):
    docs.append(Document(
        page_content=content,
        metadata={
            "kb": KB_NAME,
            "source": src_abs,
            "chunk_index": i
        }
    ))

# === 벡터스토어 준비 ===
vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding_model)

# === 같은 KB만 삭제 후 덮어쓰기 (선택) ===
try:
    vectorstore.delete(where={"kb": KB_NAME})
    print(f"[INFO] 기존 KB('{KB_NAME}') 문서 삭제 완료 (overwrite)")
except Exception as e:
    print(f"[WARN] 기존 KB 삭제 중 경고: {e}")

# === 추가 & 저장 ===
vectorstore.add_documents(docs)
# persist_directory 지정 시 자동 저장되므로 persist() 호출 불필요

print(f"[DONE] MD 인덱싱 완료: {len(docs)} chunks → {os.path.abspath(PERSIST_DIR)}")
print(f"[HINT] 검색 시 filter={{'kb': '{KB_NAME}'}} 로 MD만 조회할 수 있습니다.")
