import os
import re
import json
from typing import Dict, Any, List, Optional
from urllib.parse import unquote

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

# ------------------------
# 환경 설정
# ------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
TOURAPI_KEY    = os.getenv("TOURAPI_KEY", "")
MOBILE_OS      = os.getenv("MOBILE_OS", "WEB")
MOBILE_APP     = os.getenv("MOBILE_APP", "Gildongmu")

# TourAPI 엔드포인트 (HTTPS 우선, 실패시 HTTP 폴백 — v4.3 문서상 HTTP도 허용)  :contentReference[oaicite:2]{index=2}
BASE_URL_HTTPS = "https://apis.data.go.kr/B551011/KorService2"
BASE_URL_HTTP  = "http://apis.data.go.kr/B551011/KorService2"

# LLM은 이번 테스트에 필수는 아니지만, 향후 문장 파서로 확장할 수 있도록 유지
llm = ChatOpenAI(
    model="gpt-4o-mini", 
    temperature=0,
    openai_api_key=OPENAI_API_KEY)

# ------------------------
# 네트워크 유틸
# ------------------------
def _normalize_service_key(raw: str) -> str:
    """이중 인코딩된 서비스키(%2B, %3D%3D 등)를 원복. (2015년 이후 키는 인코딩 불필요)  :contentReference[oaicite:3]{index=3}"""
    if "%" in raw:
        try:
            return unquote(raw)
        except Exception:
            return raw
    return raw

SERVICE_KEY = _normalize_service_key(TOURAPI_KEY)

def _mk_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(total=3, backoff_factor=0.5,
                    status_forcelist=[429, 500, 502, 503, 504],
                    allowed_methods=["GET"])
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://",  HTTPAdapter(max_retries=retries))
    return s

SESSION = _mk_session()

def tourapi_params(**kwargs):
    """TourAPI 공통 파라미터: serviceKey/MobileOS/MobileApp/_type=json"""
    base = dict(serviceKey=SERVICE_KEY, MobileOS=MOBILE_OS, MobileApp=MOBILE_APP, _type="json")
    base.update({k: v for k, v in kwargs.items() if v not in [None, "", [], {}]})
    return base

def _request_with_fallback(path: str, params: dict, timeout: int = 30) -> requests.Response:
    """1차: HTTPS → SSLError 시 HTTP 폴백 (양쪽 모두 허용)  :contentReference[oaicite:4]{index=4}"""
    try:
        return SESSION.get(f"{BASE_URL_HTTPS}/{path}", params=params, timeout=timeout)
    except requests.exceptions.SSLError:
        return SESSION.get(f"{BASE_URL_HTTP}/{path}", params=params, timeout=timeout)

def tour_get(path: str, **params):
    """GET 호출 → items.item 리스트 반환(없으면 [])"""
    r = _request_with_fallback(path, tourapi_params(**params))
    r.raise_for_status()
    js = r.json()
    return js.get("response", {}).get("body", {}).get("items", {}).get("item", []) or []

# ------------------------
# TourAPI Tools (도구)
# ------------------------
@tool(
    "tour_area_code",
    description=(
        "지역코드/시군구코드 조회(areaCode2). 서울=areaCode '1'. "
        "필드: code/name. 마포구 등 시군구 코드는 areaCode=1로 조회 후 'name' 매칭.  :contentReference[oaicite:5]{index=5}"
    )
)
def tour_area_code(areaCode: Optional[str] = None,
                   numOfRows: int = 100, pageNo: int = 1) -> List[Dict[str, Any]]:
    return tour_get("areaCode2", areaCode=areaCode, numOfRows=numOfRows, pageNo=pageNo)

@tool(
    "tour_search_stay",
    description=(
        "숙박정보 조회(searchStay2). 파라미터: areaCode(지역), sigunguCode(시군구), "
        "numOfRows/pageNo. (contentTypeId=32 도메인)  :contentReference[oaicite:6]{index=6}"
    )
)
def tour_search_stay(areaCode: Optional[str] = None,
                     sigunguCode: Optional[str] = None,
                     lclsSystm1: Optional[str] = None,
                     lclsSystm2: Optional[str] = None,
                     lclsSystm3: Optional[str] = None,
                     numOfRows: int = 20, pageNo: int = 1) -> List[Dict[str, Any]]:
    return tour_get("searchStay2",
                    areaCode=areaCode, sigunguCode=sigunguCode,
                    lclsSystm1=lclsSystm1, lclsSystm2=lclsSystm2, lclsSystm3=lclsSystm3,
                    numOfRows=numOfRows, pageNo=pageNo)

@tool(
    "tour_detail_common",
    description="상세 공통(detailCommon2). 주소/좌표/개요/이미지YN 등 공통 메타.  :contentReference[oaicite:7]{index=7}"
)
def tour_detail_common(contentId: str) -> Dict[str, Any]:
    items = tour_get("detailCommon2", contentId=contentId,
                     defaultYN="Y", addrinfoYN="Y", mapinfoYN="Y", overviewYN="Y")
    return items[0] if items else {}

# ------------------------
# 간단 파서 & 헬퍼
# ------------------------
def extract_sigungu_name(user_query: str) -> str:
    """
    질의에서 'OO구' 패턴 우선 추출. 없으면 '마포구' 키워드 탐색.
    """
    m = re.search(r"([가-힣A-Za-z]+구)", user_query)
    if m:
        return m.group(1)
    if "마포" in user_query:
        return "마포구"
    return ""  # 못 찾으면 빈값

def find_sigungu_code(sigungu_name: str) -> Optional[str]:
    """
    서울(areaCode='1')의 시군구 목록을 받아 '마포구' 등 이름으로 코드 탐색.
    """
    if not sigungu_name:
        return None
    gu_list = tour_area_code.invoke({"areaCode": "1", "numOfRows": 100})
    # 이름 완전/부분 일치 허용
    for it in gu_list:
        if it.get("name") == sigungu_name or sigungu_name.replace(" ", "") in it.get("name", "").replace(" ", ""):
            return str(it.get("code"))
    return None

def pretty_print_stays(items: List[Dict[str, Any]], limit: int = 10):
    print("\n🏨 마포구 주변 숙박 추천 (상위 {}개)\n".format(min(limit, len(items))))
    for i, it in enumerate(items[:limit], 1):
        title = it.get("title", "")
        addr  = it.get("addr1", "")
        tel   = it.get("tel", "")
        cid   = it.get("contentid", "")
        img   = it.get("firstimage2") or it.get("firstimage") or ""
        print(f"{i}. {title}")
        if addr: print(f"   - 주소: {addr}")
        if tel:  print(f"   - 연락처: {tel}")
        if img:  print(f"   - 이미지: {img}")
        print(f"   - contentId: {cid}")
        # 상세 개요 1줄 보강 (선택)
        detail = tour_detail_common.invoke({"contentId": str(cid)}) or {}
        ov = (detail.get("overview") or "").strip()
        if ov:
            # 너무 길면 한 줄로 요약
            ov = re.sub(r"\s+", " ", ov)
            if len(ov) > 120:
                ov = ov[:118] + "…"
            print(f"   - 개요: {ov}")
        print("-")

# ------------------------
# 메인: 단일 질의 테스트
# ------------------------
def run_single_query(user_query: str = "마포구 주변 숙박시설 추천해줘"):
    # 1) 질의에서 구 이름 추출
    sigungu_name = extract_sigungu_name(user_query) or "마포구"
    # 2) 서울(areaCode=1), 마포구(sigunguCode=?)
    sigungu_code = find_sigungu_code(sigungu_name)
    if not sigungu_code:
        raise RuntimeError(f"시군구 코드를 찾을 수 없습니다: {sigungu_name}")

    # 3) 숙박 목록 조회 (searchStay2)
    stays = tour_search_stay.invoke({
        "areaCode": "1",             # 서울
        "sigunguCode": sigungu_code, # 마포구
        "numOfRows": 30, "pageNo": 1
    })

    if not stays:
        print("검색 결과가 없습니다. 조건을 넓혀보세요.")
        return

    # 4) 출력
    pretty_print_stays(stays, limit=10)

if __name__ == "__main__":
    # 프록시/인증서로 TLS 문제가 계속되면, 아래 환경변수를 비워두고 재시도하세요.
    # os.environ["HTTPS_PROXY"] = ""
    # os.environ["HTTP_PROXY"]  = ""
    # os.environ["REQUESTS_CA_BUNDLE"] = ""
    # os.environ["SSL_CERT_FILE"] = ""

    run_single_query("마포구 주변 숙박시설 추천해줘")