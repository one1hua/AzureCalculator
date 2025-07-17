import requests
import pandas as pd

def fetch_database_prices(region="koreacentral", max_records=500):
    prices = []
    count = 0
    url = (
        "https://prices.azure.com/api/retail/prices"
        f"?$filter=serviceFamily eq 'Databases' and armRegionName eq '{region}'"
    )

    while url and count < max_records:
        response = requests.get(url)
        data = response.json()
        items = data.get("Items", [])
        for item in items:
            prices.append(item)  # 👉 이 줄이 반드시 있어야 합니다
            count += 1
            if count >= max_records:
                break
        url = data.get("NextPageLink", None)

    return pd.DataFrame(prices)

# 실행
df_db_prices = fetch_database_prices()

# 저장
df_db_prices.to_csv("database_prices_sample.csv", index=False)
print("✅ 저장 완료: database_prices_sample.csv")

