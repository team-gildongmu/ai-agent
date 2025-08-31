import os, math, json, re, requests
from typing import Dict, Any, List, Optional, Tuple, TypedDict
from urllib.parse import unquote
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END

from googleapiclient.discovery import build

# ---------------------------------------------------------------------
# 환경설정
# ---------------------------------------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TOURAPI_KEY    = os.getenv("TOURAPI_KEY")
MOBILE_OS      = os.getenv("MOBILE_OS", "WEB")
MOBILE_APP     = os.getenv("MOBILE_APP", "Gildongmu")
BASE_URL_HTTPS = "https://apis.data.go.kr/B551011/KorService2"
BASE_URL_HTTP  = "http://apis.data.go.kr/B551011/KorService2"

# LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=OPENAI_API_KEY)

# ---------------------------------------------------------------------
# 서비스키 및 세션 설정
# ---------------------------------------------------------------------
def _normalize_service_key(raw: str) -> str:
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
    base = dict(serviceKey=SERVICE_KEY, MobileOS=MOBILE_OS, MobileApp=MOBILE_APP, _type="json")
    base.update({k: v for k, v in kwargs.items() if v not in [None, "", [], {}]})
    return base

def _request_with_fallback(path: str, params: dict, timeout: int = 30):
    try:
        print(f"[LOG] HTTPS 요청: {path} params={params}")
        return SESSION.get(f"{BASE_URL_HTTPS}/{path}", params=params, timeout=timeout)
    except requests.exceptions.SSLError:
        print(f"[WARN] HTTPS 실패 → HTTP 폴백: {path}")
        return SESSION.get(f"{BASE_URL_HTTP}/{path}", params=params, timeout=timeout)

def tour_get(path: str, **params):
    r = _request_with_fallback(path, tourapi_params(**params))
    r.raise_for_status()
    js = r.json()
    items = js.get("response", {}).get("body", {}).get("items", {}).get("item", []) or []
    print(f"[LOG] {path} 결과 건수={len(items)}")
    return items

# ---------------------------------------------------------------------
# 예시 도구 (숙박, 위치기반, 상세)
# ---------------------------------------------------------------------
@tool("tour_area_code", description="지역코드/시군구코드 조회(areaCode2).")
def tour_area_code(areaCode: Optional[str] = None,
                   numOfRows: int = 100, pageNo: int = 1) -> List[Dict[str, Any]]:
    return tour_get("areaCode2", areaCode=areaCode, numOfRows=numOfRows, pageNo=pageNo)

@tool("tour_search_stay", description="숙박정보 조회(searchStay2).")
def tour_search_stay(areaCode: Optional[str] = None, sigunguCode: Optional[str] = None,
                     numOfRows: int = 30, pageNo: int = 1) -> List[Dict[str, Any]]:
    return tour_get("searchStay2", areaCode=areaCode, sigunguCode=sigunguCode,
                    numOfRows=numOfRows, pageNo=pageNo)

@tool("tour_location_based", description="위치기반 관광정보 조회(locationBasedList2).")
def tour_location_based(mapX: float, mapY: float, radius: int = 3000,
                        contentTypeId: Optional[int] = None,
                        arrange: str = "E",
                        numOfRows: int = 30, pageNo: int = 1) -> List[Dict[str, Any]]:
    radius = min(max(100, int(radius)), 20000)
    return tour_get("locationBasedList2", mapX=mapX, mapY=mapY, radius=radius,
                    arrange=arrange, contentTypeId=contentTypeId,
                    numOfRows=numOfRows, pageNo=pageNo)

@tool("tour_detail_common", description="상세 공통(detailCommon2).")
def tour_detail_common(contentId: str) -> Dict[str, Any]:
    items = tour_get("detailCommon2", contentId=contentId,
                     defaultYN="Y", addrinfoYN="Y", mapinfoYN="Y", overviewYN="Y")
    return items[0] if items else {}

# ---------------------------------------------------------------------
# 상태와 그래프
# ---------------------------------------------------------------------
class TripState(TypedDict, total=False):
    userQuery: str
    origin: Dict[str, float]
    days: int
    mode: str
    pois: List[Dict[str, Any]]
    plan: Dict[str, Any]

SYSTEM_PARSE_PROMPT = """\
다음 문장에서 여행 일수(기본 1), 이동수단(walk/drive/transit),
성향 태그(자연/도심/야경/역사/맛집 등 1~3개), (있다면) 지역명(시/구)을 추출해 JSON으로.
문장: "{query}"
출력 예시: {{"days":1,"mode":"walk","tags":["도심","맛집"],"area": "부산 사하구"}}
"""

def robust_json(s: str, fallback: dict) -> dict:
    try:
        m = re.search(r"\{.*\}", s, re.S)
        return json.loads(m.group(0)) if m else fallback
    except Exception:
        return fallback

def node_parse_query(state: TripState):
    print(f"[LOG] node_parse_query 입력: {state['userQuery']}")
    prompt = SYSTEM_PARSE_PROMPT.format(query=state["userQuery"])
    out = llm.invoke(prompt)
    text = getattr(out, "content", out)
    parsed = robust_json(text, {"days": state.get("days", 1), "mode": state.get("mode", "walk"), "tags": []})
    print(f"[LOG] node_parse_query 결과: {parsed}")
    days = int(parsed.get("days", state.get("days", 1)) or 1)
    mode = parsed.get("mode", state.get("mode", "walk")) or "walk"
    tags = parsed.get("tags", [])
    return {**state, "days": days, "mode": mode, "tags": tags}

def node_fetch_pois(state: TripState):
    print(f"[LOG] node_fetch_pois 실행, origin=({state['origin']['mapX']}, {state['origin']['mapY']})")
    x, y = state["origin"]["mapX"], state["origin"]["mapY"]
    spots = tour_location_based.invoke({
        "mapX": x, "mapY": y, "radius": 3000, "contentTypeId": 12, "arrange": "E", "numOfRows": 5
    })
    eats = tour_location_based.invoke({
        "mapX": x, "mapY": y, "radius": 3000, "contentTypeId": 39, "arrange": "E", "numOfRows": 5
    })
    print(f"[LOG] node_fetch_pois 결과: 관광지 {len(spots)}개, 음식점 {len(eats)}개")
    pois = spots + eats
    return {**state, "pois": pois}

def node_build_itinerary(state: TripState):
    print(f"[LOG] node_build_itinerary 실행, days={state.get('days')}, pois={len(state.get('pois', []))}")
    if not state.get("pois"):
        return {**state, "plan": {"days": []}}
    days = state.get("days", 1)
    pois = state["pois"]
    plan_days = []
    per_day = max(1, len(pois) // days)
    for d in range(days):
        segs = pois[d*per_day:(d+1)*per_day]
        plan_days.append({"day": d+1, "segments": segs})
    plan = {"summary": f"{days}일 일정 추천 코스", "days": plan_days}
    return {**state, "plan": plan}

# 그래프 구성
graph = StateGraph(TripState)
graph.add_node("ParseQuery", node_parse_query)
graph.add_node("FetchPOIs", node_fetch_pois)
graph.add_node("BuildItinerary", node_build_itinerary)
graph.set_entry_point("ParseQuery")
graph.add_edge("ParseQuery", "FetchPOIs")
graph.add_edge("FetchPOIs", "BuildItinerary")
graph.add_edge("BuildItinerary", END)
app = graph.compile()

# ---------------------------------------------------------------------
# 실행 예시
# ---------------------------------------------------------------------
if __name__ == "__main__":
    init: TripState = {
        "userQuery": "내 주변에서 2박 3일 정도 여행코스 짜줘!",
        "origin": {"mapX": 126.98375, "mapY": 37.563446},
        "days": 1,
        "mode": "walk",
    }
    out = app.invoke(init)
    print("\n===== 최종 추천 결과 =====")
    print(json.dumps(out.get("plan", {}), ensure_ascii=False, indent=2))
