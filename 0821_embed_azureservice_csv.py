# ingest_csv.py
# CSV → (텍스트 조합/청크) → 임베딩 → Chroma(persist_directory)에 저장
# 사용법 예시는 파일 하단 또는 README 참고

import os
import argparse
from typing import List, Optional

import pandas as pd
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings

# 긴 텍스트 컬럼이 있을 때 자연스러운 청크를 위해 사용
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    from langchain.text_splitter import RecursiveCharacterTextSplitter


# -----------------------------
# 환경/임베딩 구성
# -----------------------------
def load_env(env_path: str) -> None:
    if os.path.exists(env_path):
        load_dotenv(env_path)


def build_embedding() -> AzureOpenAIEmbeddings:
    """
    .env 예시:
      AZURE_OPENAI_API_KEY=...
      AZURE_OPENAI_ENDPOINT=https://<리소스명>.openai.azure.com
      AZURE_OPENAI_API_VERSION=2024-12-01-preview
      AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
    """
    endpoint = "https://azure-openai-price01.openai.azure.com"  # ✅ 베이스 URL
    api_key = "8D0TqUJQKj7SfHCMcRSAs5cJBSbvvQIcsuZ6QvHFKWsVsP9JyhooJQQJ99BGACNns7RXJ3w3AAABACOG0jZk"                  # ✅ API 키
    api_version = "2024-12-01-preview"
    deployment = "text-embedding-3-small"

    return AzureOpenAIEmbeddings(
        azure_endpoint="https://azure-openai-price01.openai.azure.com",          # 반드시 "베이스 URL" 형태여야 함
        azure_deployment="text-embedding-3-small",      # 배포명
        api_key=os.environ["8D0TqUJQKj7SfHCMcRSAs5cJBSbvvQIcsuZ6QvHFKWsVsP9JyhooJQQJ99BGACNns7RXJ3w3AAABACOG0jZk"],
        openai_api_version="2024-12-01-preview",
    )


# -----------------------------
# CSV → Document 변환 로직
# -----------------------------
def infer_text_columns(df: pd.DataFrame, explicit_cols: Optional[List[str]]) -> List[str]:
    """자연어 텍스트로 임베딩할 컬럼 선택 (명시 없으면 object 타입 자동 선택)"""
    if explicit_cols:
        return [c for c in explicit_cols if c in df.columns]
    return [c for c in df.columns if df[c].dtype == "object"]


def compose_row_text(row: pd.Series, text_cols: List[str]) -> str:
    """행을 사람이 읽기 좋은 자연어 컨텍스트로 변환"""
    parts = []
    for c in text_cols:
        val = row.get(c)
        if pd.isna(val):
            continue
        parts.append(f"{c}: {val}")
    text = "\n".join(parts).strip()
    return text if text else "(empty)"


def clean_empty_strings(df: pd.DataFrame) -> pd.DataFrame:
    """공백/빈 문자열을 NaN으로 치환 (isna 판별 용이)"""
    return df.replace(r"^\s*$", pd.NA, regex=True)


# -----------------------------
# 인덱싱(임베딩) 실행
# -----------------------------
def index_csv(
    input_csv: str,
    persist_dir: str,
    kb_name: str,
    id_column: Optional[str],
    text_columns: Optional[List[str]],
    meta_columns: Optional[List[str]],
    chunk_size: int,
    chunk_overlap: int,
    encoding: Optional[str],
    overwrite: bool,
    sample: Optional[int],
):
    os.makedirs(persist_dir, exist_ok=True)

    # CSV 로드
    df = pd.read_csv(input_csv, encoding=encoding) if encoding else pd.read_csv(input_csv)
    df = clean_empty_strings(df)

    if sample and sample > 0:
        df = df.head(sample)

    text_cols = infer_text_columns(df, text_columns)
    meta_cols = [c for c in (meta_columns or []) if c in df.columns]

    if not text_cols:
        raise ValueError("임베딩에 사용할 텍스트 컬럼이 없습니다. --text-columns 옵션을 지정해 주세요.")

    # 벡터 스토어 & 임베딩 준비
    embedding = build_embedding()
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embedding)

    # 같은 KB 이름 문서만 선택적으로 삭제 후 재인덱싱
    if overwrite:
        try:
            vectorstore.delete(where={"kb": kb_name})
            print(f"[INFO] 기존 KB('{kb_name}') 문서 삭제 완료 (overwrite)")
        except Exception as e:
            print(f"[WARN] 기존 문서 삭제 중 경고: {e}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    docs, ids = [], []
    src_abs = os.path.abspath(input_csv)

    for i, row in df.iterrows():
        # page_content 생성
        content = compose_row_text(row, text_cols)

        # 메타데이터 구성
        metadata = {
            "kb": kb_name,
            "source": src_abs,
            "row_index": int(i),
        }
        for c in meta_cols:
            v = row.get(c)
            if pd.isna(v):
                continue
            metadata[c] = str(v)

        # 고유 ID
        if id_column and id_column in df.columns and not pd.isna(row[id_column]):
            base_id = f"{kb_name}:{str(row[id_column])}"
        else:
            base_id = f"{kb_name}:row:{i}"

        # 청크 분할 (짧으면 1개)
        base_doc = Document(page_content=content, metadata=metadata)
        chunks = splitter.split_documents([base_doc])

        for j, ch in enumerate(chunks):
            docs.append(ch)
            ids.append(f"{base_id}#c{j}")

    if not docs:
        print("[WARN] 추가할 문서(청크)가 없습니다.")
        return

    vectorstore.add_documents(docs, ids=ids)

    print(f"[DONE] CSV 인덱싱 완료: {len(docs)} chunks → {persist_dir}")
    print(f"[HINT] KB 이름: {kb_name} (검색 시 filter={{'kb': '{kb_name}'}} 로 활용 가능)")


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CSV → Chroma 인덱싱 스크립트")
    p.add_argument("--input", required=True, help="CSV 파일 경로 (예: ./azure_service.csv)")
    p.add_argument("--persist-dir", default="./chromaDB", help="Chroma 저장 경로 (기본: ./chromaDB)")
    p.add_argument("--kb", default="kb_csv", help="KB 이름(metadata.kb) (기본: kb_csv)")
    p.add_argument("--id-column", default=None, help="행 고유 ID로 사용할 컬럼명 (예: id, sku)")
    p.add_argument(
        "--text-columns",
        default=None,
        help="임베딩에 사용할 자연어 컬럼들(쉼표 구분). 미지정 시 object 타입 자동 선택",
    )
    p.add_argument(
        "--meta-columns",
        default=None,
        help="메타데이터로 보존할 컬럼들(쉼표 구분). 예: price,currency,region",
    )
    p.add_argument("--chunk-size", type=int, default=1000, help="청크 크기 (기본 1000)")
    p.add_argument("--chunk-overlap", type=int, default=150, help="청크 오버랩 (기본 150)")
    p.add_argument("--encoding", default=None, help="CSV 인코딩 (예: utf-8, cp949)")
    p.add_argument("--env", default=".env", help=".env 경로 (기본 ./.env)")
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="같은 KB 이름 문서를 삭제 후 재인덱싱 (중복 방지)",
    )
    p.add_argument(
        "--sample",
        type=int,
        default=None,
        help="상위 N행만 테스트 인덱싱 (성능/비용 점검용)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    load_env(args.env)

    text_cols = args.text_columns.split(",") if args.text_columns else None
    meta_cols = args.meta_columns.split(",") if args.meta_columns else None

    input_abs = os.path.abspath(args.input)
    persist_abs = os.path.abspath(args.persist_dir)

    print(f"[CFG] input={input_abs}")
    print(f"[CFG] persist_dir={persist_abs}")
    print(f"[CFG] kb={args.kb}")
    print(f"[CFG] id_column={args.id_column}")
    print(f"[CFG] text_columns={text_cols}")
    print(f"[CFG] meta_columns={meta_cols}")

    index_csv(
        input_csv=input_abs,
        persist_dir=persist_abs,
        kb_name=args.kb,
        id_column=args.id_column,
        text_columns=text_cols,
        meta_columns=meta_cols,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        encoding=args.encoding,
        overwrite=args.overwrite,
        sample=args.sample,
    )


if __name__ == "__main__":
    main()

~                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
~                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
~                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
~                                                                                                                                                  
