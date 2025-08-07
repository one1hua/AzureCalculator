import requests
import pandas as pd
import json
import os
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- 하드코딩된 디스크 티어별 성능 정보 (Azure 문서 기반) ---
# 이 정보는 Azure의 공식 문서에서 직접 확인하여 최신 값으로 업데이트해야 합니다.
# https://learn.microsoft.com/ko-kr/azure/virtual-machines/disks-types#standard-ssd
# https://learn.microsoft.com/ko-kr/azure/virtual-machines/disks-types#standard-hdd
DISK_PERFORMANCE_TIERS = {
    "Standard SSD": {
        "E1": {"GiB": 4, "IOPS": 500, "MBps": 100},
        "E2": {"GiB": 8, "IOPS": 500, "MBps": 100},
        "E3": {"GiB": 16, "IOPS": 500, "MBps": 100},
        "E4": {"GiB": 32, "IOPS": 500, "MBps": 100},
        "E6": {"GiB": 64, "IOPS": 500, "MBps": 100},
        "E10": {"GiB": 128, "IOPS": 500, "MBps": 100},
        "E15": {"GiB": 256, "IOPS": 500, "MBps": 100},
        "E20": {"GiB": 512, "IOPS": 500, "MBps": 100},
        "E30": {"GiB": 1024, "IOPS": 500, "MBps": 100},
        "E40": {"GiB": 2048, "IOPS": 500, "MBps": 100},
        "E50": {"GiB": 4096, "IOPS": 500, "MBps": 100},
        "E60": {"GiB": 8192, "IOPS": 2000, "MBps": 400},
        "E70": {"GiB": 16384, "IOPS": 4000, "MBps": 600},
        "E80": {"GiB": 32767, "IOPS": 6000, "MBps": 750},
    },
    "Standard HDD": {
        "S4": {"GiB": 32, "IOPS": 500, "MBps": 60},
        "S6": {"GiB": 64, "IOPS": 500, "MBps": 60},
        "S10": {"GiB": 128, "IOPS": 500, "MBps": 60},
        "S15": {"GiB": 256, "IOPS": 500, "MBps": 60},
        "S20": {"GiB": 512, "IOPS": 500, "MBps": 60},
        "S30": {"GiB": 1024, "IOPS": 500, "MBps": 60},
        "S40": {"GiB": 2048, "IOPS": 500, "MBps": 60},
        "S50": {"GiB": 4096, "IOPS": 500, "MBps": 60},
        "S60": {"GiB": 8192, "IOPS": 1300, "MBps": 300},
        "S70": {"GiB": 16384, "IOPS": 2000, "MBps": 500},
        "S80": {"GiB": 32768, "IOPS": 2000, "MBps": 500},
    }
}
# --- 하드코딩된 디스크 티어별 성능 정보 끝 ---


def fetch_disk_type_capabilities_from_json(file_path="vm_skus_koreacentral.json"):
    """
    Azure CLI에서 가져온 JSON 파일에서 디스크 유형별 스펙 (최대 IOPS, 대역폭, 크기 범위)을 추출합니다.
    'resourceType'이 'disks'인 항목만 처리합니다.
    """
    if not os.path.exists(file_path):
        logging.error(f"오류: '{file_path}' 파일을 찾을 수 없습니다.")
        return {}

    try:
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            sku_data = json.load(f)
    except json.JSONDecodeError as e:
        logging.error(f"오류: '{file_path}' 파일이 유효한 JSON 형식이 아닙니다. 오류: {e}")
        return {}

    disk_type_specs = {}
    for sku_item in sku_data:
        resource_type = sku_item.get("resourceType", "")
        
        if "disks" in resource_type:
            disk_type_name = sku_item.get("name", "") # 예: StandardSSD_LRS, Standard_LRS, Premium_LRS
            
            if disk_type_name and "capabilities" in sku_item:
                max_size_gib = None
                max_iops_read_write = None
                max_bandwidth_mbps_read_write = None

                for cap in sku_item["capabilities"]:
                    cap_name = cap.get("name")
                    cap_value = cap.get("value")
                    
                    if cap_name == "MaxSizeGiB":
                        try: max_size_gib = int(cap_value)
                        except (ValueError, TypeError): pass
                    elif cap_name == "MaxIOpsReadWrite":
                        try: max_iops_read_write = int(cap_value)
                        except (ValueError, TypeError): pass
                    elif cap_name == "MaxBandwidthMBpsReadWrite":
                        try: max_bandwidth_mbps_read_write = int(cap_value)
                        except (ValueError, TypeError): pass
                
                # 디스크 유형 스펙 저장
                disk_type_specs[disk_type_name] = {
                    "MaxGiB_Type": max_size_gib,
                    "MaxIOPS_Type": max_iops_read_write,
                    "MaxMBps_Type": max_bandwidth_mbps_read_write
                }
    logging.info(f"JSON 파일에서 {len(disk_type_specs)}개의 디스크 유형 스펙을 로드했습니다.")
    return disk_type_specs

def fetch_and_filter_disk_prices_with_specs(region="koreacentral", sku_json_file="vm_skus_koreacentral.json"):
    prices = []
    
    logging.info("디스크 유형 스펙 로드 중...")
    disk_type_specs_map = fetch_disk_type_capabilities_from_json(sku_json_file)
        
    count = 0
    url = (
        "https://prices.azure.com/api/retail/prices"
        f"?$filter=armRegionName eq '{region}' and serviceFamily eq 'Storage' and serviceName eq 'Storage'"
    )

    while url:
        logging.info(f"가격 데이터 페이지 가져오는 중... (현재 {count}개)")
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            logging.error(f"API 요청 중 오류 발생: {e}")
            break

        data = resp.json()
        items = data.get("Items", [])

        for item in items:
            sku_name_retail = item.get("skuName", "") # 예: E50 LRS, S4 LRS
            product_name = item.get("productName", "").lower() # 예: Standard SSD Managed Disks
            price_type = item.get("type", "")
            reservation_term = item.get("reservationTerm", "None")
            meter_name = item.get("meterName", "").lower()
            arm_sku_name = item.get("armSkuName", "") # 예: Standard_SSD_E50_LRS

            # --- 1차 필터링: Standard SSD 또는 Standard HDD만 포함 ---
            is_standard_ssd = "standard ssd" in product_name or "standard ssd" in meter_name
            is_standard_hdd = "standard hdd" in product_name or "standard hdd" in meter_name
            
            if not (is_standard_ssd or is_standard_hdd):
                # logging.debug(f"DEBUG: Not SSD/HDD: {product_name} / {meter_name}")
                continue

            # --- 2차 필터링: Consumption 또는 Reservation (1년, 3년) ---
            if not (price_type == "Consumption" or (price_type == "Reservation" and reservation_term in ["1 Year", "3 Year"])):
                # logging.debug(f"DEBUG: Not target price type: {price_type} / {reservation_term}")
                continue

            # --- 3차 필터링: Backup Disk, Snapshot, Data Transfer 등 제외 ---
            # LRS만 뽑는 조건도 여기에 추가. skuName_retail, meter_name, arm_sku_name 모두 "LRS"를 포함하는지 확인
            if (
                "backup disk" in meter_name or 
                "snapshot" in meter_name or 
                "data transfer" in meter_name or
                "bursting" in meter_name or # 버스팅 비용 제외 (일반 디스크 가격 아님)
                "iops" in meter_name or     # IOPS 트랜잭션 비용 제외 (디스크 자체 가격 아님)
                "transactions" in meter_name or # 트랜잭션 비용 제외
                "disk operations" in meter_name or # 디스크 작업 비용 제외
                "write operations" in meter_name or # 쓰기 작업 비용 제외
                "confidential compute encryption" in meter_name.lower() or   # <-- 이 줄 추가
                # LRS만 포함하는 조건 추가
                not (" lrs" in sku_name_retail.lower() or " lrs" in arm_sku_name.lower()) # skuName_retail 또는 armSkuName에 ' LRS' 포함 확인
            ):
                # logging.debug(f"DEBUG: Filtered out by specific exclusions or not LRS: SKU={sku_name_retail}, Meter={meter_name}, ArmSKU={arm_sku_name}")
                continue
            
            # --- 디스크 유형 및 티어 식별 ---
            disk_category = "Standard SSD" if is_standard_ssd else "Standard HDD"
            
            # skuName_retail에서 티어 정보 추출 (예: E50 LRS -> E50)
            # ProductName (예: Standard SSD E1) 에서도 티어 정보를 추출할 수 있음.
            tier = ""
            # skuName_retail에서 " LRS" 부분 제거하고 티어만 추출
            if " lrs" in sku_name_retail.lower():
                tier = sku_name_retail.lower().replace(" lrs", "").upper().strip()
            elif " " in sku_name_retail: # 혹시 LRS가 없는 경우, 마지막 단어를 티어로
                tier = sku_name_retail.split(" ")[-1].upper().strip()
            else: # E50, S4 같이 단독으로 오는 경우
                tier = sku_name_retail.upper().strip()


            # --- 스펙 정보 가져오기 ---
            provisioned_gib = 'N/A'
            provisioned_iops = 'N/A'
            provisioned_mbps = 'N/A'

            # 하드코딩된 티어별 성능 정보 매핑 시도
            if disk_category in DISK_PERFORMANCE_TIERS and tier in DISK_PERFORMANCE_TIERS[disk_category]:
                tier_info = DISK_PERFORMANCE_TIERS[disk_category][tier]
                provisioned_gib = tier_info.get("GiB", 'N/A')
                provisioned_iops = tier_info.get("IOPS", 'N/A')
                provisioned_mbps = tier_info.get("MBps", 'N/A')
            else:
                logging.debug(f"DEBUG: 티어 성능 정보 없음: {disk_category} - {tier}. SKU: {sku_name_retail}")


            # az vm list-skus (또는 az disk list-skus)에서 얻은 일반적인 디스크 유형 스펙
            # armSkuName_full (예: Standard_SSD_E50_LRS)에서 디스크 유형 (StandardSSD_LRS) 추출
            disk_type_for_lookup = None
            if arm_sku_name: # Standard_SSD_E50_LRS -> Standard_SSD_LRS
                parts = arm_sku_name.split('_')
                if len(parts) >= 3 and (parts[-1] == 'LRS' or parts[-1] == 'ZRS'): # LRS/ZRS 등으로 끝나는 경우
                    disk_type_for_lookup = '_'.join(parts[:-2]) + '_' + parts[-1] # Standard_SSD_LRS
                elif "Standard_SSD" in arm_sku_name: # E1, E10 같은 티어 없는 경우 대비
                    disk_type_for_lookup = "StandardSSD_LRS"
                elif "Standard" in arm_sku_name and "HDD" not in arm_sku_name:
                    disk_type_for_lookup = "Standard_LRS" # Standard HDD는 주로 이렇게 명명됨


            generic_specs = {}
            if disk_type_for_lookup and disk_type_for_lookup in disk_type_specs_map:
                generic_specs = disk_type_specs_map.get(disk_type_for_lookup, {})
            else:
                logging.debug(f"DEBUG: 일반 디스크 유형 스펙 매핑 실패: {disk_type_for_lookup}. ArmSKU: {arm_sku_name}")


            prices.append({
                "DiskCategory": disk_category,
                "Tier": tier,
                "SKUName_Retail": sku_name_retail, # E50 LRS, S4 LRS
                "PriceType": price_type,
                "ReservationTerm": reservation_term,
                "RetailPrice": item.get("retailPrice"),
                "CurrencyCode": item.get("currencyCode"),
                "UnitOfMeasure": item.get("unitOfMeasure"),
                "Location": item.get("armRegionName"),
                "Provisioned_GiB": provisioned_gib, 
                "Provisioned_IOPS": provisioned_iops,
                "Provisioned_MBps": provisioned_mbps,
                "MeterName": item.get("meterName"),
            })
            count += 1

        url = data.get("NextPageLink", None)

    df = pd.DataFrame(prices)
    df = df.fillna('N/A') 
    
    output_filename = f"azure_disk_prices_filtered_specs_koreacentral.csv"
    df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    logging.info(f"✅ 저장 완료: {output_filename}")
    logging.info(f"총 {count}개의 디스크 가격 및 스펙 데이터를 가져왔습니다.")

# --- 실행 부분 ---
if __name__ == "__main__":
    fetch_and_filter_disk_prices_with_specs(region="koreacentral", sku_json_file="vm_skus_koreacentral.json")