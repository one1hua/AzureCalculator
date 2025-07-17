import requests
import pandas as pd
import time


def fetch_compute_prices(region='koreacentral', currency='USD'):
    all_items = []
    base_url = "https://prices.azure.com/api/retail/prices"
    filter_query = (
        f"?$filter=armRegionName eq '{region}' "
        f"and serviceFamily eq 'Compute' "
        f"and currencyCode eq '{currency}'"
    )
    next_url = base_url + filter_query

    while next_url:
        print(f"📡 Fetching: {next_url}")

        success = False
        retries = 0
        while not success and retries < 3:
            try:
                response = requests.get(next_url, timeout=30)
                response.raise_for_status()
                data = response.json()
                items = data.get("Items", [])
                all_items.extend(items)
                next_url = data.get("NextPageLink", None)
                success = True
                time.sleep(0.2)
            except requests.exceptions.RequestException as e:
                retries += 1
                print(f"⚠️ 요청 실패, 재시도 {retries}/3 - {e}")
                time.sleep(2)

        if not success:
            print("❌ 3회 시도 실패. 다음 페이지로 넘어갑니다.")
            next_url = None

    print(f"✅ 총 {len(all_items)}개 항목 수집 완료!")
    return pd.DataFrame(all_items)


def filter_compute_services(df):
    print(f"✅ 초기 행 수: {len(df)}")

    # 누락 컬럼 채워 넣기
    for col in ['priceType', 'reservationTerm', 'serviceName', 'skuName']:
        if col not in df.columns:
            df[col] = ''

    # NaN 채우기 → 필터에서 다 날아가는 것 방지
#    df['priceType'] = df['priceType'].fillna('')
#    df['reservationTerm'] = df['reservationTerm'].fillna('')
#    df['serviceName'] = df['serviceName'].fillna('')
#    df['skuName'] = df['skuName'].fillna('')

    # ① 온디맨드 + 예약 가격 필터
#    df = df[df['priceType'].isin(['Consumption', 'Reservation'])]
#    df = df[df['reservationTerm'].isin(['', '1 Year', '3 Years'])]
#    print(f"➡️ 가격 타입+예약 기간 필터 후: {len(df)}")

    # ② VM만 D/F/N/H 시리즈 필터
    is_vm = df['serviceName'] == 'Virtual Machines'
    is_dfn_h = df['skuName'].str.startswith(('D', 'F', 'N', 'H'))
    is_other = df['serviceName'] != 'Virtual Machines'

    filtered_df = df[(is_vm & is_dfn_h) | is_other]
    print(f"➡️ 최종 필터 후 행 수: {len(filtered_df)}")

    return filtered_df


def clean_columns(df):
    cols = [
        'armRegionName', 'serviceFamily', 'serviceName',
        'skuName', 'armSkuName', 'retailPrice',
        'priceType', 'reservationTerm', 'unitOfMeasure', 'currencyCode'
    ]
    for col in cols:
        if col not in df.columns:
            df[col] = ''
    return df[cols]


def main():
    print("✅ Azure Retail Prices API에서 한국 리전 Compute 서비스 가격 가져오기 시작!")

    # 1️⃣ API 크롤링
    prices_df = fetch_compute_prices(region='koreacentral')

    # 2️⃣ 필터링
    filtered_df = filter_compute_services(prices_df)

    # 3️⃣ 컬럼 정리
    final_df = clean_columns(filtered_df)

    # 4️⃣ CSV 저장
    final_df.to_csv('azure_korea_compute_prices_filtered.csv', index=False)
    print("✅ CSV 저장 완료: azure_korea_compute_prices_filtered.csv")


if __name__ == "__main__":
    main()

