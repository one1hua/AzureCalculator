from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
import os

# Azure OpenAI 설정
AZURE_API_KEY = "8D0TqUJQKj7SfHCMcRSAs5cJBSbvvQIcsuZ6QvHFKWsVsP9JyhooJQQJ99BGACNns7RXJ3w3AAABACOG0jZk"
AZURE_ENDPOINT = "https://azure-openai-price01.openai.azure.com/openai/deployments/text-embedding-3-small/embeddings?api-version=2023-05-15"
DEPLOYMENT_NAME = "text-embedding-3-small"
API_VERSION = "2024-02-01"

embedding_model = AzureOpenAIEmbeddings(
    azure_endpoint=AZURE_ENDPOINT,
    azure_deployment=DEPLOYMENT_NAME,
    api_key=AZURE_API_KEY,
    openai_api_version=API_VERSION
)

# Markdown 파일 로드
with open("./azurepricecalculate_description.md", "r", encoding="utf-8") as f:
    text = f.read()

# 텍스트를 청크로 분할
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_text(text)
print(f"총 {len(texts)}개의 청크로 분할됨.")

# from_texts로 임베딩 및 Chroma DB 저장
vectorstore = Chroma.from_texts(
    texts=texts,
    embedding=embedding_model,
    persist_directory="./chromaDB"
)

print("✅ 임베딩 완료")
