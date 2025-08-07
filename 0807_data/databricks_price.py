import requests
import pandas as pd
import logging
from typing import Dict, Any

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def get_databricks_prices(region: str) -> pd.DataFrame:
    """
    Azure Databricks의 가장 많이 사용되는 DBU 가격을 추출하여 DataFrame으로 반환합니다.
    (Standard/Premium All-purpose Compute, Standard/Premium Jobs Compute)
    """
    prices = []
    
    # API 필터: Databricks 서비스, 종량제, DBU 포함
    url = (
        "https://prices.azure.com/api/retail/prices"
        f"?$filter=armRegionName eq '{region}' and serviceName eq 'Azure Databricks' and type eq 'Consumption' and contains(meterName, 'DBU')"
    )

    logging.info(f"Azure Databricks 가격 데이터 가져오는 중... (리전: {region})")
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

    # B2B 고객에게 가장 중요한 4가지 DBU 유형을 필터링
    filtered_df = df[
        (df['metername'].str.contains('all-purpose compute dbu', case=False) & df['metername'].str.contains('standard', case=False)) |
        (df['metername'].str.contains('all-purpose compute dbu', case=False) & df['metername'].str.contains('premium', case=False)) |
        (df['metername'].str.contains('jobs compute dbu', case=False) & df['metername'].str.contains('standard', case=False)) |
        (df['metername'].str.contains('jobs compute dbu', case=False) & df['metername'].str.contains('premium', case=False))
    ].copy()

    # 불필요한 Free Trial 항목 제거
    filtered_df = filtered_df[~filtered_df['metername'].str.contains('free', case=False, na=False)]

    if not filtered_df.empty:
        # 최종 결과에 필요한 컬럼만 선택
        result_df = filtered_df[[
            'metername', 'retailprice', 'unitofmeasure', 'currencycode'
        ]]
        return result_df.sort_values(by='retailprice').reset_index(drop=True)
    else:
        return pd.DataFrame()

# --- 실행 부분 ---
if __name__ == "__main__":
    region_name = "koreacentral"
    databricks_df = get_databricks_prices(region=region_name)
    
    if not databricks_df.empty:
        output_filename = f"databricks_prices_{region_name}.csv"
        databricks_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
        print("="*50)
        print(f"✅ Azure Databricks 가격이 '{output_filename}' 파일로 저장되었습니다.")
        print("포함된 항목: Standard/Premium All-purpose & Jobs Compute DBU")
        print(databricks_df)
        print("="*50)
    else:
        print("="*50)
        print("⚠️ Azure Databricks 가격을 찾을 수 없거나 데이터가 비어 있습니다.")
        print("="*50)
