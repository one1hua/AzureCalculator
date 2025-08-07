import requests
import pandas as pd
import logging
from typing import List, Any

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def get_representative_aoai_prices(region: str) -> pd.DataFrame:
    """
    사용자가 지정한 대표적인 Azure OpenAI 모델들의 가격을 추출합니다.
    """
    prices = []
    
    # API 필터: serviceName이 'Azure OpenAI'인 종량제(Consumption) 가격만 가져옴
    url = (
        "https://prices.azure.com/api/retail/prices"
        f"?$filter=armRegionName eq '{region}' and productName eq 'Azure OpenAI' and type eq 'Consumption'"
    )

    logging.info(f"대표 모델 가격 데이터 가져오는 중... (리전: {region})")
    while url:
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            logging.error(f"API 요청 중 오류 발생: {e}")
            return pd.DataFrame()

        data = resp.json()
        items = data.get("Items", [])
        prices.extend(items)
        url = data.get("NextPageLink", None)
    
    if not prices:
        return pd.DataFrame()

    df = pd.DataFrame(prices)
    df = df.fillna('N/A')
    df.columns = df.columns.str.lower()
    df['retailprice'] = pd.to_numeric(df['retailprice'], errors='coerce')

    # 사용자가 지정한 대표 모델의 meterName 키워드
    representative_keywords = [
        'gpt-4o-0806', 'gpt-4o-mini-0718', 'text-embedding-ada-002'
    ]

# 필터링 로직: meterName에 키워드가 포함된 행만 선택하고, 'Batch'가 없는 행만 남김
    filtered_df = df[
        df['metername'].str.contains('|'.join(representative_keywords), case=False, na=False)
    ].copy()

    # 'Batch'가 포함된 항목을 명시적으로 제거
    filtered_df = filtered_df[~filtered_df['metername'].str.contains('Batch', case=False, na=False)]
    
    # Free, Dedicated 항목 제거
    filtered_df = filtered_df[~filtered_df['productname'].str.contains('free|dedicated', case=False, na=False)]
    
    if filtered_df.empty:
        return pd.DataFrame()
    result_df = filtered_df[[
        'productname', 'metername', 'retailprice', 'unitofmeasure', 'currencycode'
    ]]
    return result_df.sort_values(by=['productname', 'metername']).reset_index(drop=True)

# --- 실행 부분 ---
if __name__ == "__main__":
    region_name = "koreacentral"
    rep_aoai_df = get_representative_aoai_prices(region=region_name)
    
    if not rep_aoai_df.empty:
        output_filename = f"representative_aoai_prices_{region_name}.csv"
        rep_aoai_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
        print("="*50)
        print(f"✅ 대표 Azure OpenAI 모델 가격이 '{output_filename}' 파일로 저장되었습니다.")
        print(rep_aoai_df)
        print("="*50)
    else:
        print("="*50)
        print("⚠️ 대표 Azure OpenAI 가격을 찾을 수 없거나 데이터가 비어 있습니다.")
        print("="*50)