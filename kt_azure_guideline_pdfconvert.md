MS Azure 원가산정 참고  
 
 
< Cloud사업담당 2025.04, v1> 
□ 원가산정 개요 
○ Cloud 원가 산정 
- MS Azure 등 Cloud는 단순 개별 서비스(서버, 스토리지, 네트워크 등) 제공이 아닌  
효율적인 Architecture를 설계/적용하고 세부 서비스를 구성해서 고객의 업무에 활용하는 특성 
- 고객이 Cloud를 활용하는 목적, 요구사항을 기반으로 단계적으로 원가 산정 진행 
1) 고객 요구사항을 기반으로 기술 요건을 세부 정의  
2) 기술 요건을 만족하는 Cloud 아키텍처 및 개별 서비스를 선정 및 자원 구성 
3) 서비스 시나리오에 따른 트래픽 등 사용량을 예측하고 이에 따른 원가를 추정해서 산정 
- Cloud 요금은 실제 서비스들을 사용한 만큼 발생 
. Cloud는 사전에 모든 요금이 확정되지 않고, 일부는 사용량에 따라 비용 추가 또는 감소 발생  
→ 실제 사용요금은 익월 초 확정 
e.g. outbound 트래픽 요금, 로드밸런서 데이터 처리량, VM 추가생성, 주말에 VM(종량제)을 꺼두는 등 
□ Cloud 아키텍처 개요 
○ Cloud 아키텍처 개요 
- Well-Architected Framework  https://learn.microsoft.com/ko-kr/azure/well-architected/ 
 
- 모두가 Cloud SASolutions Architect가 될 필요는 없으나 1) 개요를 이해하고, 2) 간단한 사례는 다룰 수 있어야 함 


---

 
                                                                2/ 7 
○ 간단한 Cloud 아키텍처 구성 사례 
- Azure Architecture Center https://learn.microsoft.com/en-us/azure/architecture/ 
- (예시) N-Tier Architecture (aka. 3-tier Architecture – Web, WAS, DB) 
* https://medium.com/@shbelay/design-vnet-architecture-6e131a2fe13e 
 
*IP Address of Azure Resources 
 
*DNAT Rule in the Firewall Instance 
 
Firewall 
Public: 172.179.92.140  
Private: 10.0.3.4 
Web Tier Load Balancer 
10.0.0.4 
Web Tier VM Scale Set 
10.0.0.5, 10.0.0.6 
Application Tier Load Balancer 
10.0.1.4 
Application Tier VM Scale Set 
10.0.1.5, 10.0.1.6 
SQL Servers 
10.0.2.0/24 
Source 
* (any) 
Protocol 
TCP 
Destination 
172.179.92.140 
Destination Port 
80 
Translated Address 
10.0.0.4 
Translated Port 
80 
○ 개별 서비스 선정 및 자원 구성 
- 기존의 서버 & 네트워크 장비등을 구성하는 방식(legacy)과 유사하게, 가상화 된 VM 등 Cloud 자원 구성 
. ex. [서버 → Virtual Machine],    [L2 Switch → Subnet],    [L3 Connectivity → VNET Peering 등] 
- DB 등의 경우, VM에 패키지 설치 및 구성도 가능하나, 관리형 서비스도 활용 가능 
. 다양한 관리기능 제공 - 복제(Replication) 구성, 서버 구성 또는 Serverless 방식, 백업 
- Serverless 방식 구성 
. 사용자는 비즈니스 로직에 집중, 나머지 관리는 거의 대부분 자동으로 수행 
        [서비스 및 구성 방식 비교]   *serverless: 서버를 신경 쓰지 않음 
VM 방식 
(고객이 서버를 VM으로 전환 및 관리) 
관리형 서비스 
(Cloud에서 일부 관리기능 제공) 
Serverless* 
(사용자는 비즈니스 로직에 집중) 
 
 
 
비용 
비효율적 
성능 
제약 
쉽게 적용 
비용 
효율적 
성능제약 
일부 해소 
쉽게 적용 
비용 
효율적 
성능제약 
거의 해소 
동작방식 
이해 필요 


---

 
                                                                3/ 7 
□ Cloud 자원 구성 옵션 및 원가 산정 
○ 서비스와 자원 
- 서비스는 자원을 구성하고 생성하는 기능, e.g. VM 서비스 → VM 인스턴스 자원 
- 서비스 설명: Azure 홈페이지 참고 
- 서비스 구성 및 요금: https://azure.microsoft.com/en-us/pricing/calculator/ 
○ Virtual Machine https://learn.microsoft.com/ko-kr/azure/virtual-machines/ 
 
 
ㅇ VM 유형(Type) 
https://azure.microsoft.com/en-us/pricing/details/virtual-
machines/series/ 
- Category, Instance Series, Instance 
- 개발용: A-series 
- 버스트(저렴한 가격): Bs-Series 
- 범용: D-Series              *뒤에 숫자가 클수록 최신 세대 
. Dv5   – Intel 3세대 Ice Lake 
. Dasv5 – AMD 3세대 EPYC Milan 
- 컴퓨팅 최적화: F-Series  
. Fsv2  – Intel 2세대 SkyLake 
ㅇ 요금제 
- 무약정 Pay as you go 
- 약정(1년 또는 3년) 
. Reservations: VM 자체를 약정 
중간에 비싼 VM으로 변경 가능 
. Savings Plan:: 컴퓨팅 비용 지출을 약정,  
금액을 채울 수 있는 다른 자원으로 변경 가능 
ㅇ Disk, Bandwidth 
- VM 약정 시 원가계산을 정확하게 하기 위해  
별도 서비스로 구분해서 산정 
- VM Type에 따른 vCPU와 메모리 비율 matching 
.  각 Type 별로 vCPU 개수와 메모리 용량(in GB) 비율이 고정됨 → 요구사항에 따라 다른 Type을 활용 대응 
. ex.  ▪ D-Series – 1 : 4     ▪ F-Series – 1 : 2 
- GPU VM 
. 현재 국내기준 H100 및 T4, P40 정도 수급가능, A100은 단종 예정으로 공급이 거의 불가능 (미리 확인 필수) 
Instance Series 
세부 옵션 
vCPU 
메모리(GiB) 
GPU 
GPU 
메모리(GiB) 
월 요금 
NDsr H100 v5 
*5년약정 옵션 
ND H100 v5 
96 
1900 
8 
80 
$96,894.36 
NDada H100 v5 
Standard_NC40ads_H100_v5 
40 
320 
1 
94 
$6,878.79 
Standard_NC80adis_H100_v5 
80 
640 
2 
188 
$13,757.58 
- 원가산정 참고: 약정 요금제는 매입할인 없음, 종량제의 경우 10% 할인 적용 
. VM에 약정 요금제(RI/SP) 를 적용하더라도 Disk나 Bandwidth는 약정 적용이 안 되므로 분리해서 원가 계산 


---

 
                                                                4/ 7 
○ Managed Disks https://learn.microsoft.com/ko-kr/azure/virtual-machines/managed-disks-overview 
- 용도에 따라 유형 및 세부 서비스 옵션(IOPS, Throughput)을 선택 가능 
. Premium SSD를 권장, 가격 절감 시 Standard SSD, 데이터 저장 용도로 Premium SSD v2 옵션 활용 
구분 
Ultra Disk 
프리미엄 SSD v2 
프리미엄 SSD 
표준 SSD 
표준 HDD 
유형 
SSD 
SSD 
SSD 
SSD 
HDD 
용도 
IO 집약적 워크로드 
- SAP HANA, 최상위 
계층 데이터베이스 
(예: SQL, Oracle) 및 
다른 트랜잭션 집약적 
워크로드 
짧은 대기 시간과 
높은 IOPS 및 
처리량이 지속적으로 
요구되는 프로덕션 
및 성능에 민감한 
워크로드 
프로덕션 및 
성능이 중요한 
워크로드 
웹 서버, 조금 
사용되는 
엔터프라이즈 
애플리케이션 
및 개발/테스트 
백업, 
중요하지 
않음, 가끔 
액세스 
최대크기 
65,536GiB 
65,536GiB 
32,767GiB 
32,767GiB 
32,767GiB 
Througput 
10,000 MB/s 
1,200MB/s 
900MB/s 
750MB/s 
500MB/s 
최대 IOPS 
400,000 
80,000 
20,000 
6,000 
2,000, 
3,000* 
OS 디스크? 
아니요 
아니요 
예 
예 
예 
가격 예시 
256GB 
$287.49 
(3,000 IOPS, 125 MB/S) 
$23.36 
(3,000 IOPS, 125 MB/S) 
$38.01 
(1,100 IOPS, 125 MB/s) 
$19.40 
(500 IOPS, 100 MB/s) 
$11.38 
* Managed Disks 는 약정 옵션이 없으므로 VM 약정 시 Disk 가격만 별도로 10% 할인된 원가로 계산 필요 
* 상기 옵션(IOPS, Throughput 등)은 최적 값이 아니며, 업무 환경이나 기존 기준을 참고하여 조정 필요  
○ Public IP Addresses https://learn.microsoft.com/en-us/azure/virtual-network/ip-services/public-ip-addresses 
- 외부에서 접속 가능한 IP 주소 
. VM network interface, VM scale sets, Azure Load Balancers(public), VPN/ER Gateway, NAT gateway, 
Application Gateway, Azure Firewalls, Bastion Hosts, Route Servers, API Management 등 에 할당 가능 
- (참고) 간단한 네트워크 구성 예시 및 요금 
 
Type 
Basic (ARM) 
Standard (ARM) 
Global (ARM) 
Dynamic IPv4 address 
$0.004/hour 
N/A 
N/A 
Static IPv4 address 
$0.0036/hour 
$0.005/hour 
$0.01/hour 
Public IPv4 prefix2 
N/A 
$0.006 per IP/hour 
$0.012 per IP/hour 
 


---

 
                                                                5/ 7 
○ Azure Database for PostgreSQL https://learn.microsoft.com/ko-kr/azure/postgresql/ 
* 다양한 Database 서비스 중에 PostgreSQL 사례 Review 
- Single server vs. Flexible server 
. Single server는 곧 단종, Flexible server는 다양하고 유연한 옵션 제공 
 
- Flexible server의 다양한 관리 기능(fully managed) 예시 
. 가용성 영역(AZ) 지정 옵션                *가용성 영역: Region을 구성하는 여러 데이터센터 중 하나의 물리적 센터(장애 격리)  
. 고가용성 구성 – 서로 다른 AZ Primary 및 Standby 서버를 구성 
. 자동 패치와 maintenance window 지정 관리 
. 자동 백업 (Zone Redundant Storage (ZRS)), 기본 7일 ~ 최대 35일 
. 성능을 높일 수 있도록 서버 사양 재 구성 (중지 및 재시작 필요) 
. 커넥션 pool 관리, 자동 HA 구성 및 Failover 처리 - Primary 노드 fail 시, Standby 자동 승격 처리 
 
 
- Pricing Calculator 예시 
 


---

 
                                                                6/ 7 
 
 
VM과 유사하게, 서비스에 약정 요금제 옵션(Reservations) 설정 시 디스크 요금만 10% 할인 적용 필요 
 
○ Azure OpenAI https://learn.microsoft.com/ko-kr/azure/ai-services/openai/overview 
- 구축 사례 (외부에서 연동) 
 
 
ㅇ 가격 옵션 
- Pay as you go: 사용한 토큰 단위로 과금 
- PTU (Provisioned Throughput Unit): 일정한 처리량을 약정 요금제로 사용 
*1PTU: 분당 2500토큰 처리 용량 
* PTU 방식에서 약정(reserved) 요금 선택 시 PAYG 대비 70% 이상 비용 절감 
ㅇ Deployment 옵션 
- Global Deployment: 국내에서 서빙을 하더라도 자원 부족 시 해외 자원 활용 가능, 
  최소 15PTU, 5 PTU 단위 증가 
- Regional Deployment: 국내 자원으로 서빙, 최소 50 PTU, 50 PTU 단위 증가 
 
 
 
 
 


---

 
                                                                7/ 7 
□ 원가 산정 
○ 원가산정 예시 
- 기본규칙: 무약정(PAYG) 요금제로 생성한 자원은 매입할인 10% 반영, 약정 요금제 자원은 매입할인 없음 
자원 
사양 
요금제 
List Price 
원가 
(할인율) 
VM#1 
D4v5 4 vCPU, 16 GB RAM 
PAYG 
$172.28  
$155.05  
10% 
Standard SSD E15 256GiB 
PAYG 
$19.20  
$17.28  
10% 
VM#2 
D4v5 4 vCPU, 16 GB RAM 
Reserved(1yr) 
$101.67  
$101.67  
0% 
Standard SSD E15 256GiB 
PAYG 
$19.20  
$17.28  
10% 
Public IP 
Standard (ARM) Static IP 
PAYG 
$3.65  
$3.29  
10% 
AOAI 
GPT-4o-Regional-API-128K 50 PTUs 
Reserved(1mo) 
$13,000  
$13,000  
0% 
Traffic 
Internet Egress (Public Internet) 
PAYG 
$99  
$89.10  
10% 
○ 유의사항 
- Cloud는 사용한 만큼 요금 발생, Pay-As-You-Go    (cf. 약정 요금제 자원은 고정된 요금 발생) 
. VM(PAYG) 요금을 아끼기 위해 주말에 꺼 놓은 것은 흔한 운영방식 
. 트래픽 등 요금은 고객이 통제할 수 없는 영역 - 이벤트 발생 시 예상 못한 트래픽 급증 등 
- 견적서에 ‘본 견적이 최종 요금을 확정하는 것은 아니며, 실제 사용량에 따라 요금은 변동될 수 있습니다.’ 
제약조건 명기 
□ VDC 수지분석 반영 
○ 수지분석 원가 반영 
- 물품 공급에 Azure 원가 반영  
 
□ Summary 
• MS Azure 등 Cloud는 단순 개별 서비스(서버, 스토리지, 네트워크 등) 제공이 아닌 효율적인 Architecture를 
설계/적용하고 세부 서비스를 구성해서 고객의 업무에 활용하는 특성 
• 원가산정 참고: 약정 요금제는 매입할인 없음, 종량제의 경우 10% 할인 적용 
• 기본규칙: 무약정(PAYG) 요금제로 생성한 자원은 매입할인 10% 반영, 약정 요금제 자원은 매입할인 없음 
• 견적서에 ‘본 견적이 최종 요금을 확정하는 것은 아니며, 실제 사용량에 따라 요금은 변동될 수 있습니다.’ 
제약조건 명기 
 


---

