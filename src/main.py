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
# TourAPI Tools
# ---------------------------------------------------------------------
@tool("tour_area_code", description="지역코드/시군구코드 조회(areaCode2).")
def tour_area_code(areaCode: Optional[str] = None,
                   numOfRows: int = 100, pageNo: int = 1) -> List[Dict[str, Any]]:
    return tour_get("areaCode2", areaCode=areaCode, numOfRows=numOfRows, pageNo=pageNo)

@tool("tour_area_based", description="지역기반 관광정보 조회(areaBasedList2).")
def tour_area_based(areaCode: Optional[str] = None, sigunguCode: Optional[str] = None,
                    contentTypeId: Optional[int] = None, arrange: str = "C",
                    numOfRows: int = 10, pageNo: int = 1) -> List[Dict[str, Any]]:
    return tour_get("areaBasedList2", areaCode=areaCode, sigunguCode=sigunguCode,
                    contentTypeId=contentTypeId, arrange=arrange,
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

@tool("tour_search_stay", description="숙박정보 조회(searchStay2).")
def tour_search_stay(areaCode: Optional[str] = None, sigunguCode: Optional[str] = None,
                     numOfRows: int = 5, pageNo: int = 1) -> List[Dict[str, Any]]:
    return tour_get("searchStay2", areaCode=areaCode, sigunguCode=sigunguCode,
                    numOfRows=numOfRows, pageNo=pageNo)

@tool("tour_detail_common", description="상세 공통(detailCommon2).")
def tour_detail_common(contentId: str) -> Dict[str, Any]:
    items = tour_get("detailCommon2", contentId=contentId,
                     defaultYN="Y", addrinfoYN="Y", mapinfoYN="Y", overviewYN="Y")
    return items[0] if items else {}

# ---------------------------------------------------------------------
# Google Enricher Tools (구조화 + 이미지)
# ---------------------------------------------------------------------
@tool(
    "google_enrich",
    description="Google Custom Search로 보강. 타이틀/스니펫/링크를 반환해 LLM 설명/이유/키워드 생성에 사용."
)
def google_enrich(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    service = build("customsearch", "v1", developerKey=os.getenv("GOOGLE_API_KEY"))
    res = service.cse().list(q=query, cx=os.getenv("SEARCH_ENGINE_ID"), num=num_results, lr="lang_ko").execute()
    out = []
    for it in res.get("items", []):
        out.append({
            "title": it.get("title", ""),
            "snippet": it.get("snippet", ""),
            "link": it.get("link", "")
        })
    return out

@tool(
    "google_image",
    description="Google Custom Search 이미지 검색. 질의에 맞는 대표 이미지 URL 목록을 반환."
)
def google_image(query: str, num_results: int = 1) -> List[str]:
    service = build("customsearch", "v1", developerKey=os.getenv("GOOGLE_API_KEY"))
    res = service.cse().list(q=query, cx=os.getenv("SEARCH_ENGINE_ID"), searchType="image", num=num_results, safe="active", lr="lang_ko").execute()
    return [it.get("link") for it in res.get("items", []) if it.get("link")]

# ---------------------------------------------------------------------
# 상태와 그래프
# ---------------------------------------------------------------------
class TripState(TypedDict, total=False):
    userQuery: str
    origin: Dict[str, float]
    days: int
    mode: str
    area: Optional[str]
    areaCode: Optional[str]
    sigunguCode: Optional[str]
    tags: List[str]
    areaResolved: bool
    pois: List[Dict[str, Any]]
    stays: List[Dict[str, Any]]
    plan: Dict[str, Any]

SYSTEM_PARSE_PROMPT = """\
다음 문장에서 여행 일수(기본 1), 이동수단(walk/drive/transit),
성향 태그(자연/도심/야경/역사/맛집/힐링/신나게 등 1~3개), (있다면) 지역명(시/군/구)을 추출해 JSON으로.
문장: "{query}"
출력 예시: {{"days":3,"mode":"walk","tags":["힐링","자연"],"area":"부천"}}
"""

def robust_json(s: str, fallback: dict) -> dict:
    try:
        m = re.search(r"\{.*\}", s, re.S)
        return json.loads(m.group(0)) if m else fallback
    except Exception:
        return fallback

# -----------------------------
# 유틸: 이미지 보강
# -----------------------------
def fetch_image_for(title: str) -> str:
    try:
        imgs = google_image.invoke({"query": f"{title} 사진"})
        return imgs[0] if imgs else ""
    except Exception as e:
        print(f"[WARN] 이미지 검색 실패: {title}, {e}")
        return ""

def attach_images_if_missing(items: List[Dict[str, Any]]):
    for it in items:
        img = it.get("firstimage") or it.get("firstimage2") or it.get("image")
        if not img:
            img = fetch_image_for(it.get("title", ""))
            if img:
                it["image"] = img

# -----------------------------
# 지역명 → areaCode/sigunguCode/좌표 변환 (안전)
# -----------------------------
def resolve_area_to_origin(area_name: str) -> Tuple[Optional[float], Optional[float], Optional[str], Optional[str]]:
    """
    지역명 → (mapx, mapy, areaCode, sigunguCode)
    - 시군구 이름(부천/마포 등) 우선 정밀 매칭
    - 매칭 실패 시 (None, None, None, None) 반환 (절대 임의 좌표로 대체하지 않음)
    """
    if not area_name:
        return None, None, None, None

    print(f"[LOG] 지역명 '{area_name}' → 행정코드 탐색")
    areas = tour_area_code.invoke({"areaCode": None, "numOfRows": 100})
    norm = area_name.replace("시", "").replace("군", "").replace("구", "").replace(" ", "")

    # 모든 시/도에 대해 시군구 전체를 훑어서 정밀 탐색
    for a in areas:
        ac = str(a.get("code"))
        sggs = tour_area_code.invoke({"areaCode": ac, "numOfRows": 500})
        for s in sggs:
            sname = (s.get("name") or "").replace("시","" ).replace("군","" ).replace("구","" ).replace(" ", "")
            if not sname:
                continue
            if sname == norm or norm in sname or sname in norm:
                sigungu_code = str(s.get("code"))
                # 대표 POI 1개로 좌표 추정 (실 지오코딩 대체)
                pois = tour_area_based.invoke({"areaCode": ac, "sigunguCode": sigungu_code, "contentTypeId": 12, "numOfRows": 1})
                if pois:
                    try:
                        return float(pois[0]["mapx"]), float(pois[0]["mapy"]), ac, sigungu_code
                    except Exception:
                        pass
                return None, None, ac, sigungu_code

    print("[WARN] 행정코드 매핑 실패 → origin 변경하지 않음")
    return None, None, None, None

# -----------------------------
# 노드들
# -----------------------------
def node_parse_query(state: TripState):
    print(f"[LOG] node_parse_query 입력: {state['userQuery']}")
    prompt = SYSTEM_PARSE_PROMPT.format(query=state["userQuery"])
    out = llm.invoke(prompt)
    text = getattr(out, "content", out)
    parsed = robust_json(text, {"days": state.get("days", 1), "mode": state.get("mode", "walk"), "tags": [], "area": None})

    # '내 주변/근처' 표현이 있으면 지역 해석을 생략하고 origin 유지
    q = state["userQuery"]
    if any(tok in q for tok in ["내 주변", "내주변", "근처", "가까운", "주변"]):
        parsed["area"] = None
        print("[LOG] '내 주변/근처' 감지 → area 해석 생략, origin 유지")

    print(f"[LOG] node_parse_query 결과: {parsed}")
    days = int(parsed.get("days", state.get("days", 1)) or 1)
    mode = parsed.get("mode", state.get("mode", "walk")) or "walk"
    tags = parsed.get("tags", [])
    area = parsed.get("area")
    return {**state, "days": days, "mode": mode, "tags": tags, "area": area}


def node_resolve_area(state: TripState):
    area_name = state.get("area")
    if not area_name:
        print("[LOG] 지역명 없음 또는 '내 주변' → 기존 origin 사용")
        state["areaResolved"] = True  # 주변 기준 사용을 성공으로 간주
        return state

    mx, my, ac, sc = resolve_area_to_origin(area_name)
    if ac or sc:
        state["areaCode"], state["sigunguCode"] = ac, sc
        if mx is not None and my is not None:
            state["origin"] = {"mapX": mx, "mapY": my}
            print(f"[LOG] 지역 좌표 반영: ({mx}, {my}), areaCode={ac}, sigunguCode={sc}")
        else:
            print(f"[LOG] 코드만 매칭됨(areaCode={ac}, sigunguCode={sc}) → origin은 유지")
        state["areaResolved"] = True
        return state

    print("[LOG] 지역 매핑 실패 → origin 유지 + 구글 보강 모드로 전환")
    state["areaResolved"] = False
    return state


def _build_google_only_candidates(state: TripState) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """매핑 실패 시: 전부 Google 검색으로 후보 생성 (POI/MEAL, STAY). 좌표는 미제공(None)."""
    theme = " ".join(state.get("tags", [])) or "여행"
    area_kw = state.get("area") or "내 주변"

    poi_res = google_enrich.invoke({"query": f"{area_kw} {theme} 명소 추천"})
    eat_res = google_enrich.invoke({"query": f"{area_kw} 맛집 추천"})
    stay_res = google_enrich.invoke({"query": f"{area_kw} 호텔 숙소 추천"})

    def to_item(r: Dict[str, str], seg_type: str) -> Dict[str, Any]:
        img = fetch_image_for(r.get("title", ""))
        return {
            "type": seg_type,
            "title": r.get("title"),
            "desc": r.get("snippet"),
            "reason": f"TourAPI 매핑 실패로 Google 검색 결과를 기반으로 추천했습니다.",
            "image": img,
            "coords": None,
            "source": r.get("link")
        }

    pois_meals = [to_item(r, "POI") for r in poi_res[:8]] + [to_item(r, "MEAL") for r in eat_res[:8]]
    stays = [
        {
            "title": r.get("title"),
            "desc": r.get("snippet"),
            "reason": "사용자 선택용 숙박 후보(구글 보강)",
            "image": fetch_image_for(r.get("title", "")),
            "coords": None,
            "source": r.get("link")
        }
        for r in stay_res[:5]
    ]
    return pois_meals, stays


def node_fetch_pois(state: TripState):
    # 매핑 실패 시: TourAPI를 전혀 사용하지 않고 Google+LLM만 사용
    if state.get("areaResolved") is False:
        print("[LOG] areaResolved=False → Google-only 후보 생성")
        pois, stays = _build_google_only_candidates(state)
        print(f"[LOG] Google-only 후보: POI/MEAL {len(pois)}개, STAY {len(stays)}개")
        return {**state, "pois": pois, "stays": stays}

    print(f"[LOG] node_fetch_pois 실행, origin=({state['origin']['mapX']}, {state['origin']['mapY']})")
    x, y = state["origin"]["mapX"], state["origin"]["mapY"]

    spots = tour_location_based.invoke({"mapX": x, "mapY": y, "radius": 7000, "contentTypeId": 12, "arrange": "E", "numOfRows": 20})
    eats  = tour_location_based.invoke({"mapX": x, "mapY": y, "radius": 7000, "contentTypeId": 39, "arrange": "E", "numOfRows": 20})
    stays = tour_location_based.invoke({"mapX": x, "mapY": y, "radius": 7000, "contentTypeId": 32, "arrange": "E", "numOfRows": 10})

    # 이미지 보강
    attach_images_if_missing(spots)
    attach_images_if_missing(eats)
    attach_images_if_missing(stays)

    pois = spots + eats

    if not pois:
        # TourAPI에 데이터가 없을 때도 Google-only로 보강
        print(f"[LOG] TourAPI 결과 없음 → Google-only 후보 생성")
        pois, stays_google = _build_google_only_candidates(state)
        # stays는 둘 중 더 풍부한 쪽 사용
        if not stays:
            stays = stays_google

    print(f"[LOG] node_fetch_pois 결과: 관광지+음식점 {len(pois)}개, 숙박 {len(stays)}개")
    return {**state, "pois": pois, "stays": stays}


def node_build_itinerary(state: TripState):
    print(f"[LOG] node_build_itinerary 실행, days={state.get('days')}, tags={state.get('tags')}, pois={len(state.get('pois', []))}, stays={len(state.get('stays', []))}")
    if not state.get("pois"):
        return {**state, "plan": {"days": []}}

    # LLM에게 일정 설계 위임 (태그까지 반영)
    pois_text = json.dumps(state["pois"][:50], ensure_ascii=False)
    stays_text = json.dumps(state["stays"][:8], ensure_ascii=False)

    prompt = f"""
    사용자의 여행 질의: {state['userQuery']}
    성향 태그: {state.get('tags', [])}
    총 {state['days']}일 일정입니다.

    - 후보(관광지/음식점) JSON:
    {pois_text}
    - 숙박 후보 JSON:
    {stays_text}

    코스 구성 규칙:
    - 1일 코스: 숙박은 포함하지 않아도 됨.
    - 2일 이상: 숙박은 하루 일정에 넣지 말고, 별도로 stays 리스트로만 제공(사용자 선택).
    - 매일 관광지(POI)는 2~3곳, 음식점(MEAL)은 점심/저녁으로 1~2곳 배치하되, 가까운 동선 위주로 묶기.
    - 각 항목은 title, desc, reason, image(가능 시), coords(mapx,mapy가 있으면 사용/없으면 null)를 포함.
    - 최종 JSON 키: keywords(5~10개), days(일자별 segments 배열), stays(숙박 후보 배열), summary(3~4문장 통합 설명)
    - days의 segment.type은 "POI" 또는 "MEAL" 중 하나로.
    - 한국어로.
    """

    out = llm.invoke(prompt)
    plan = robust_json(getattr(out, "content", out), {"keywords": [], "days": [], "stays": [], "summary": ""})

    # LLM이 stays를 비웠다면, TourAPI/Google 후보로 보완
    if not plan.get("stays") and state.get("stays"):
        simple_stays = []
        for s in state["stays"][:5]:
            simple_stays.append({
                "title": s.get("title"),
                "desc": s.get("desc") or "주변 숙박 후보",
                "reason": s.get("reason") or "일정 지역과 접근성이 좋아 추천",
                "image": s.get("image") or s.get("firstimage") or s.get("firstimage2"),
                "coords": s.get("coords") or {"mapx": s.get("mapx"), "mapy": s.get("mapy")}
            })
        plan["stays"] = simple_stays

    return {**state, "plan": plan}

# 그래프 구성
graph = StateGraph(TripState)
graph.add_node("ParseQuery", node_parse_query)
graph.add_node("ResolveArea", node_resolve_area)
graph.add_node("FetchPOIs", node_fetch_pois)
graph.add_node("BuildItinerary", node_build_itinerary)
graph.set_entry_point("ParseQuery")
graph.add_edge("ParseQuery", "ResolveArea")
graph.add_edge("ResolveArea", "FetchPOIs")
graph.add_edge("FetchPOIs", "BuildItinerary")
graph.add_edge("BuildItinerary", END)
app = graph.compile()

# ---------------------------------------------------------------------
# 실행 예시
# ---------------------------------------------------------------------
if __name__ == "__main__":
    init: TripState = {
        "userQuery": "부천에서 힐링 테마로 2박 3일 여행코스 추천해줘",
        "origin": {"mapX": 126.98375, "mapY": 37.563446},  # 초기값(명동) — 지역 해석 실패 시 그대로 유지
        "days": 3,
        "mode": "walk",
    }
    out = app.invoke(init)
    print("\n===== 최종 추천 결과(JSON) =====")
    print(json.dumps(out.get("plan", {}), ensure_ascii=False, indent=2))
