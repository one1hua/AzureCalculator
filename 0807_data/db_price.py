import requests
import pandas as pd

def fetch_filtered_db_compute_prices(region="koreacentral"):
    prices = []
    url = f"https://prices.azure.com/api/retail/prices?$filter=serviceFamily eq 'Databases' and armRegionName eq '{region}' and skuName eq 'vCore'"

    while url:
        response = requests.get(url)
        data = response.json()
        items = data.get("Items", [])
        prices.extend(items)
        url = data.get("NextPageLink", None)

    df = pd.DataFrame(prices)

    # 필터 조건 정의
    mysql_product = "Azure Database for MySQL Flexible Server General Purpose Series Compute"
    postgre_product = "Azure Database for PostgreSQL Flexible Server General Purpose Dsv3 Series Compute"
    sqldb_prodcut = "SQL Database Single/Elastic Pool General Purpose - Compute Gen5"

    df_filtered = df[
        (df["serviceName"].isin(["Azure Database for MySQL", "Azure Database for PostgreSQL", "SQL Database"])) &
        (df["productName"].isin([mysql_product, postgre_product, sqldb_prodcut]))
    ]
    
    # 필요한 열만 추출
    df_selected = df_filtered[[
        "currencyCode",
        "retailPrice",
        "armRegionName",
        "productName",
        "skuName",
        "serviceName",
        "serviceFamily",
        "unitOfMeasure",
    "type",
    "reservationTerm"
    ]]

    return df_selected
# 실행 및 저장
df_db_compute = fetch_filtered_db_compute_prices()
df_db_compute.to_csv("filtered_db_compute_prices.csv", index=False)
print("✅ 저장 완료: filtered_db_compute_prices.csv")

