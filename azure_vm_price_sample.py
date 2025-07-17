import requests
import pandas as pd

def fetch_virtual_machine_prices(region="koreacentral", max_records=500):
    prices = []
    count = 0
    url = (
        "https://prices.azure.com/api/retail/prices"
        f"?$filter=serviceFamily eq 'Compute' and armRegionName eq '{region}'"
    )
    
    while url and count < max_records:
        response = requests.get(url)
        data = response.json()
        items = data.get("Items", [])
        for item in items:
            if "virtual machine" in item.get("productName", "").lower():
                prices.append(item)
                count += 1
                if count >= max_records:
                    break
        url = data.get("NextPageLink", None)

    return pd.DataFrame(prices)

# 실행
df_vm_prices = fetch_virtual_machine_prices()

# 저장하고 싶다면 아래 사용
df_vm_prices.to_csv("virtual_machine_prices_sample.csv", index=False)

