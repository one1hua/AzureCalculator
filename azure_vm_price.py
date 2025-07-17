import requests
import pandas as pd

def fetch_filtered_vm_prices(region="koreacentral"):
    prices = []
    count = 0
    url = (
        "https://prices.azure.com/api/retail/prices"
        f"?$filter=armRegionName eq '{region}' and serviceFamily eq 'Compute' and serviceName eq 'Virtual Machines'"
    )

    while url:
        resp = requests.get(url)
        data = resp.json()
        items = data.get("Items", [])
        for item in items:
            product_name = item.get("productName", "").lower()
            sku_name = item.get("skuName", "")
            price_type = item.get("type", "")
            if (
                any(sku_name.startswith(prefix) for prefix in ["D", "F", "N", "H"])
                and not sku_name.endswith("Low Priority")
                and not sku_name.endswith("Spot")
            ):
                os_type = "Windows" if "windows" in product_name else "Linux"
                prices.append({
                    "currencyCode": item.get("currencyCode"),
                    "retailPrice": item.get("retailPrice"),
                    "location": item.get("armRegionName"),
                    "skuName": sku_name,
                    "OS": os_type,
                    "type": price_type,
                    "reservationTerm": item.get("reservationTerm")
                })
                count += 1

        url = data.get("NextPageLink", None)

    df = pd.DataFrame(prices)
    df.to_csv("azure_vm_prices_filtered.csv", index=False)
    print("✅ 저장 완료: azure_vm_prices_filtered.csv")

fetch_filtered_vm_prices()

