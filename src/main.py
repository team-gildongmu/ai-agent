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
# 로깅/스위치
# ---------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()  # DEBUG/INFO/WARN/ERROR

def _log(level: str, msg: str):
    order = {"DEBUG": 10, "INFO": 20, "WARN": 30, "ERROR": 40}
    if order[level] >= order.get(LOG_LEVEL, 20):
        print(f"[{level}] {msg}")

# SSL/폴백 상태
SSL_WARNED = False                 # HTTPS→HTTP 경고는 최초 1회만
FORCE_HTTP = False                 # True면 이후 요청 전부 HTTP로만
SSL_ERRORS = 0                     # SSL 오류 누적
SSL_ERROR_THRESHOLD = int(os.getenv("SSL_ERROR_THRESHOLD", "2"))  # N회 초과 시 Google-only
GOOGLE_ONLY_SESSION = False        # 세션 전역 Google-only 전환 스위치

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
    retries = Retry(total=3, backoff_factor=0.4,
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

# ---------------------------------------------------------------------
# HTTP 요청 (HTTPS 1회 경고, 임계치 초과 시 Google-only 전환)
# ---------------------------------------------------------------------
def _request_with_fallback(path: str, params: dict, timeout: int = 20):
    global SSL_WARNED, FORCE_HTTP, SSL_ERRORS, GOOGLE_ONLY_SESSION

    if SSL_ERRORS >= SSL_ERROR_THRESHOLD:
        GOOGLE_ONLY_SESSION = True
        _log("WARN", "SSL 오류 임계 초과 → 세션 Google-only 전환")

    try_https = not FORCE_HTTP
    if try_https:
        try:
            _log("DEBUG", f"HTTPS 요청: {path} params={params}")
            return SESSION.get(f"{BASE_URL_HTTPS}/{path}", params=params, timeout=timeout)
        except requests.exceptions.SSLError:
            SSL_ERRORS += 1
            if not SSL_WARNED:
                _log("WARN", f"HTTPS 실패 → HTTP 폴백 (누적 SSL 오류 {SSL_ERRORS})")
                SSL_WARNED = True
            FORCE_HTTP = True  # 이후엔 계속 HTTP 사용

    _log("DEBUG", f"HTTP 요청: {path} params={params}")
    return SESSION.get(f"{BASE_URL_HTTP}/{path}", params=params, timeout=timeout)

def tour_get(path: str, **params):
    if GOOGLE_ONLY_SESSION:
        raise RuntimeError("GOOGLE_ONLY_SESSION")
    r = _request_with_fallback(path, tourapi_params(**params))
    r.raise_for_status()
    js = r.json()
    items = js.get("response", {}).get("body", {}).get("items", {}).get("item", []) or []
    _log("INFO", f"{path} 결과 건수={len(items)}")
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
# 상태와 그래프 (★ 대화형 필드 확장)
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

    # 대화형 확장
    history: List[Dict[str, str]]           # [{"role":"user"/"assistant","content":"..."}]
    prefs: Dict[str, Any]                    # 누적 선호/제약
    must_include: List[str]                  # 반드시 포함 키워드/지명
    exclude: List[str]                       # 제외 키워드/지명
    last_area_key: Optional[str]             # 캐시 키(지역코드 기준)
    catalog_cache: Dict[str, Any]            # {"<areaKey>":{"spots":[...],"eats":[...],"stays":[...]}}

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

# ---------------------------------------------------------------------
# 유틸: 이미지 보강
# ---------------------------------------------------------------------
def fetch_image_for(title: str) -> str:
    try:
        imgs = google_image.invoke({"query": f"{title} 사진"})
        return imgs[0] if imgs else ""
    except Exception as e:
        _log("WARN", f"이미지 검색 실패: {title}, {e}")
        return ""

def attach_images_if_missing(items: List[Dict[str, Any]]):
    for it in items:
        img = it.get("firstimage") or it.get("firstimage2") or it.get("image")
        if not img and it.get("title"):
            it["image"] = fetch_image_for(it.get("title"))

# ---------------------------------------------------------------------
# 지역명 → areaCode/sigunguCode/좌표 변환
# ---------------------------------------------------------------------
def resolve_area_to_origin(area_name: str) -> Tuple[Optional[float], Optional[float], Optional[str], Optional[str]]:
    if not area_name:
        return None, None, None, None

    _log("INFO", f"지역명 '{area_name}' → 행정코드 탐색")
    try:
        areas = tour_area_code.invoke({"areaCode": None, "numOfRows": 100})
    except Exception as e:
        _log("WARN", f"areaCode2 호출 실패: {e}")
        return None, None, None, None

    norm = area_name.replace("시", "").replace("군", "").replace("구", "").replace(" ", "")

    for a in areas:
        ac = str(a.get("code"))
        try:
            sggs = tour_area_code.invoke({"areaCode": ac, "numOfRows": 300})
        except Exception as e:
            _log("WARN", f"areaCode2 시군구 조회 실패(ac={ac}): {e}")
            continue

        for s in sggs:
            sname = (s.get("name") or "").replace("시","" ).replace("군","" ).replace("구","" ).replace(" ", "")
            if not sname:
                continue
            if sname == norm or norm in sname or sname in norm:
                sc = str(s.get("code"))
                # 좌표 추정 1회만 수행
                try:
                    pois = tour_area_based.invoke({"areaCode": ac, "sigunguCode": sc, "contentTypeId": 12, "numOfRows": 1})
                    if pois:
                        return float(pois[0]["mapx"]), float(pois[0]["mapy"]), ac, sc
                except Exception as e:
                    _log("WARN", f"좌표 추정 실패(ac={ac},sc={sc}): {e}")
                return None, None, ac, sc

    _log("WARN", "행정코드 매핑 실패")
    return None, None, None, None

# ---------------------------------------------------------------------
# 노드들
# ---------------------------------------------------------------------
def node_parse_query(state: TripState):
    _log("INFO", f"node_parse_query: {state['userQuery']}")
    out = llm.invoke(SYSTEM_PARSE_PROMPT.format(query=state["userQuery"]))
    text = getattr(out, "content", out)
    parsed = robust_json(text, {"days": state.get("days", 1), "mode": state.get("mode", "walk"), "tags": [], "area": None})

    # '내 주변/근처'이면 지역 해석 생략하고 origin 유지
    q = state["userQuery"]
    if any(tok in q for tok in ["내 주변", "내주변", "근처", "가까운", "주변"]):
        parsed["area"] = None
        _log("INFO", "'내 주변/근처' 감지 → area 해석 생략, origin 유지")

    days = int(parsed.get("days", state.get("days", 1)) or 1)
    mode = parsed.get("mode", state.get("mode", "walk")) or "walk"
    tags = parsed.get("tags", [])
    area = parsed.get("area")
    return {**state, "days": days, "mode": mode, "tags": tags, "area": area}

# --------------------- (★ 추가) 후속 지시 파서 ----------------------------
FOLLOWUP_PARSE_PROMPT = """\
아래 문장은 기존 여행 계획에 대한 '수정 지시'다. 한국어 지시에서 다음 필드를 JSON으로 채워라.
- days_delta: 일수 변경(정수, 상대/절대 모두 허용. 예: "2일로 줄여줘" => 2(절대))
- area_override: 지역명 변경이 있으면 문자열, 없으면 null
- add_tags: 추가 태그 목록
- remove_tags: 제거 태그 목록
- must_include: 반드시 포함할 지명/키워드 목록
- exclude: 제외 지명/키워드 목록
- indoor_priority: 실내/우천 대비 요청이면 true/false
- budget_per_night: 숙박 1박 예산(만원 단위 숫자) 또는 null
- notes: 기타 메모 1~2문장

문장: "{query}"
출력 예시: {{"days_delta":2,"area_override":null,"add_tags":["힐링"],"remove_tags":[],"must_include":["망원한강"],"exclude":["롯데월드"],"indoor_priority":true,"budget_per_night":12,"notes":"비 예보 대비"}}
"""

def parse_followup(query: str) -> Dict[str, Any]:
    # 1) 규칙기반 빠른 훑기
    days_abs = None
    m = re.search(r'(\d+)\s*일', query)
    if m:
        try:
            days_abs = int(m.group(1))
        except:
            pass

    indoor = any(k in query for k in ["실내", "우천", "비 오", "비가", "비와", "비와서", "비 와"])
    add_tags, remove_tags = [], []
    for kw in ["자연","도심","야경","역사","맛집","힐링","신나게","카페","전시","시장","산책","한강"]:
        if (kw in query) and ("빼" not in query and "제외" not in query):
            add_tags.append(kw)
    for kw in ["자연","도심","야경","역사","맛집","힐링","신나게","카페","전시","시장","산책","한강"]:
        if (kw in query) and ("빼" in query or "제외" in query):
            remove_tags.append(kw)

    def extract_list(q, verb_set):
        pat = r'([가-힣A-Za-z0-9\s]+)\s*(?:은|는)?\s*(?:' + '|'.join(verb_set) + r')'
        hit = re.findall(pat, q)
        items = []
        for h in hit:
            items += [w.strip() for w in re.split(r'[,\s]+', h) if w.strip()]
        return list(dict.fromkeys(items))
    musts = extract_list(query, ["넣어줘","꼭 넣어줘","포함","가야","가고 싶"])
    excls = extract_list(query, ["빼줘","제외","가지 말","안 가","안가"])

    bud = None
    m = re.search(r'(\d+)\s*만\s*원?\s*(?:이하|정도|쯤)?', query)
    if m:
        bud = int(m.group(1))

    # 2) LLM 보정(선택)
    try:
        out = llm.invoke(FOLLOWUP_PARSE_PROMPT.format(query=query))
        js = robust_json(getattr(out, "content", out), {})
    except Exception:
        js = {}

    return {
        "days_abs": js.get("days_delta") if js.get("days_delta") else days_abs,
        "area_override": js.get("area_override"),
        "add_tags": list(dict.fromkeys((js.get("add_tags") or []) + add_tags)),
        "remove_tags": list(dict.fromkeys((js.get("remove_tags") or []) + remove_tags)),
        "must_include": list(dict.fromkeys((js.get("must_include") or []) + musts)),
        "exclude": list(dict.fromkeys((js.get("exclude") or []) + excls)),
        "indoor_priority": bool(js.get("indoor_priority") or indoor),
        "budget_per_night": js.get("budget_per_night") if js.get("budget_per_night") else bud,
        "notes": js.get("notes") or ""
    }

def merge_lists(base: List[str], add: List[str], remove: List[str]) -> List[str]:
    s = {x for x in base}
    for a in add: s.add(a)
    for r in remove:
        if r in s: s.remove(r)
    return list(s)

def node_incorporate_followup(state: TripState):
    q = state.get("userQuery","").strip()
    if not q:
        return state

    state.setdefault("history", [])
    state.setdefault("prefs", {})
    state.setdefault("must_include", [])
    state.setdefault("exclude", [])
    state.setdefault("catalog_cache", {})

    state["history"].append({"role":"user","content":q})

    fu = parse_followup(q)

    if fu.get("days_abs"):
        state["days"] = max(1, int(fu["days_abs"]))

    if fu.get("area_override"):
        state["area"] = fu["area_override"]
        state["last_area_key"] = None

    state["tags"] = merge_lists(state.get("tags", []), fu.get("add_tags", []), fu.get("remove_tags", []))

    state["must_include"] = list(dict.fromkeys(state.get("must_include", []) + fu.get("must_include", [])))
    state["exclude"] = [x for x in state.get("exclude", []) if x not in fu.get("must_include", [])]
    state["exclude"] = list(dict.fromkeys(state["exclude"] + fu.get("exclude", [])))

    prefs = state["prefs"]
    if fu.get("indoor_priority") is not None:
        prefs["indoor_priority"] = fu["indoor_priority"]
    if fu.get("budget_per_night") is not None:
        prefs["budget_per_night"] = fu["budget_per_night"]
    if fu.get("notes"):
        prefs["notes"] = fu["notes"]
    state["prefs"] = prefs

    state["areaResolved"] = state.get("areaResolved", False)
    return state

# --------------------- (★ 추가) 후보 필터/캐시 ----------------------------
def _area_key(state: TripState) -> Optional[str]:
    ac, sc = state.get("areaCode"), state.get("sigunguCode")
    return f"{ac}:{sc}" if (ac or sc) else None

def _filter_by_prefs(items: List[Dict[str, Any]], state: TripState) -> List[Dict[str, Any]]:
    musts = [m.lower() for m in state.get("must_include", [])]
    excls = [e.lower() for e in state.get("exclude", [])]
    indoor = bool(state.get("prefs", {}).get("indoor_priority"))

    def is_indoor_like(it):
        title = (it.get("title") or "").lower()
        cat = str(it.get("contenttypeid") or "")
        return any(k in title for k in ["전시","박물관","미술관","갤러리","쇼핑몰","아울렛","카페","도서관","아쿠아리움"]) or cat in ["14","38","39"]

    def keep(it):
        t = (it.get("title") or "").lower()
        if excls and any(x in t for x in excls):
            return False
        if musts and any(x in t for x in musts):
            return True
        if indoor and not is_indoor_like(it):
            return False
        return True

    out = [it for it in items if keep(it)]
    for m in musts:
        if not any(m in (x.get("title","").lower()) for x in out):
            cand = next((x for x in items if m in (x.get("title","").lower())), None)
            if cand: out.append(cand)
    return out

def node_resolve_area(state: TripState):
    area_name = state.get("area")
    if not area_name:
        _log("INFO", "지역명 없음 → 기존 origin 사용")
        state["areaResolved"] = True
        return state

    mx, my, ac, sc = resolve_area_to_origin(area_name)
    if ac or sc:
        state["areaCode"], state["sigunguCode"] = ac, sc
        if mx is not None and my is not None:
            state["origin"] = {"mapX": mx, "mapY": my}
            _log("INFO", f"지역 좌표 반영: ({mx},{my}), areaCode={ac}, sigunguCode={sc}")
        else:
            _log("INFO", f"코드만 매칭됨(areaCode={ac}, sigunguCode={sc}) → origin 유지")
        state["areaResolved"] = True
        return state

    _log("WARN", "지역 매핑 실패 → origin 유지 + Google-only")
    state["areaResolved"] = False
    return state

# 최소 요구치
MIN_POI = 6
MIN_MEAL = 4
MIN_STAY = 3

def _build_google_only_candidates(state: TripState) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    theme = " ".join(state.get("tags", [])) or "여행"
    area_kw = state.get("area") or "내 주변"

    poi_res  = google_enrich.invoke({"query": f"{area_kw} {theme} 명소 추천"})
    eat_res  = google_enrich.invoke({"query": f"{area_kw} 맛집 추천"})
    stay_res = google_enrich.invoke({"query": f"{area_kw} 호텔 숙소 추천"})

    def to_item(r: Dict[str, str], seg_type: str) -> Dict[str, Any]:
        img = fetch_image_for(r.get("title", ""))
        return {
            "type": seg_type,
            "title": r.get("title"),
            "desc": r.get("snippet"),
            "reason": "Google 검색 결과를 기반으로 추천",
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
        } for r in stay_res[:5]
    ]
    return pois_meals, stays

def node_fetch_pois(state: TripState):
    # 지역 매핑 실패 또는 세션 전환 시: Google-only
    if state.get("areaResolved") is False or GOOGLE_ONLY_SESSION:
        _log("INFO", "Google-only 후보 생성 (매핑 실패 또는 세션 전환)")
        pois, stays = _build_google_only_candidates(state)
        pois = _filter_by_prefs(pois, state)
        stays = _filter_by_prefs(stays, state)
        attach_images_if_missing(pois); attach_images_if_missing(stays)
        _log("INFO", f"Google-only 후보: POI/MEAL {len(pois)}개, STAY {len(stays)}개")
        return {**state, "pois": pois, "stays": stays}

    # ---- 캐시 사용 시도
    akey = _area_key(state)
    cache = state.get("catalog_cache", {})
    if akey and akey in cache:
        _log("DEBUG", f"카탈로그 캐시 히트: {akey}")
        spots, eats, stays = cache[akey]["spots"], cache[akey]["eats"], cache[akey]["stays"]
    else:
        _log("INFO", f"node_fetch_pois origin=({state['origin']['mapX']}, {state['origin']['mapY']})")
        x, y = state["origin"]["mapX"], state["origin"]["mapY"]
        try:
            spots = tour_location_based.invoke({"mapX": x, "mapY": y, "radius": 7000, "contentTypeId": 12, "arrange": "E", "numOfRows": 20})
            eats  = tour_location_based.invoke({"mapX": x, "mapY": y, "radius": 7000, "contentTypeId": 39, "arrange": "E", "numOfRows": 20})
            stays = tour_location_based.invoke({"mapX": x, "mapY": y, "radius": 7000, "contentTypeId": 32, "arrange": "E", "numOfRows": 10})
        except Exception as e:
            _log("WARN", f"TourAPI 호출 실패 → Google-only: {e}")
            pois, stays = _build_google_only_candidates(state)
            pois = _filter_by_prefs(pois, state)
            stays = _filter_by_prefs(stays, state)
            attach_images_if_missing(pois); attach_images_if_missing(stays)
            return {**state, "pois": pois, "stays": stays}

        attach_images_if_missing(spots); attach_images_if_missing(eats); attach_images_if_missing(stays)
        if akey:
            cache[akey] = {"spots": spots, "eats": eats, "stays": stays}
            state["catalog_cache"] = cache

    # 부족 시 부분 Google 보강
    if len(spots) < MIN_POI or len(eats) < MIN_MEAL or len(stays) < MIN_STAY:
        _log("INFO", "카테고리별 부족 → 부분 Google 보강")
        g_pois, g_stays = _build_google_only_candidates(state)
        g_spots = [x for x in g_pois if x.get("type") == "POI"]
        g_eats  = [x for x in g_pois if x.get("type") == "MEAL"]
        if len(spots) < MIN_POI:
            spots += g_spots[: (MIN_POI - len(spots))]
        if len(eats) < MIN_MEAL:
            eats  += g_eats[: (MIN_MEAL - len(eats))]
        if len(stays) < MIN_STAY:
            stays += g_stays[: (MIN_STAY - len(stays))]

    # 선호 필터 적용
    spots = _filter_by_prefs(spots, state)
    eats  = _filter_by_prefs(eats, state)
    stays = _filter_by_prefs(stays, state)

    pois = spots + eats
    _log("INFO", f"최종 후보: 관광지+음식점 {len(pois)}개, 숙박 {len(stays)}개")
    return {**state, "pois": pois, "stays": stays}

def node_build_itinerary(state: TripState):
    _log("INFO", f"node_build_itinerary: days={state.get('days')}, tags={state.get('tags')}, pois={len(state.get('pois', []))}, stays={len(state.get('stays', []))}")
    if not state.get("pois"):
        return {**state, "plan": {"days": []}}

    pois_text  = json.dumps(state["pois"][:50], ensure_ascii=False)
    stays_text = json.dumps(state["stays"][:8], ensure_ascii=False)

    # 누적 선호
    musts = state.get("must_include", [])
    excls = state.get("exclude", [])
    prefs = state.get("prefs", {})
    indoor = prefs.get("indoor_priority", False)
    budget = prefs.get("budget_per_night")  # 만원

    prompt = f"""
    사용자의 여행 질의(가장 최근): {state['userQuery']}
    대화 히스토리 요약(최신 우선 3~4턴): {json.dumps(state.get('history', [])[-6:], ensure_ascii=False)}
    누적 성향 태그: {state.get('tags', [])}
    누적 필수 포함: {musts}
    누적 제외: {excls}
    실내 우선: {indoor}
    숙박 1박 예산(만원): {budget if budget else "제한없음"}
    총 {state['days']}일 일정입니다.

    - 후보(관광지/음식점) JSON:
    {pois_text}
    - 숙박 후보 JSON:
    {stays_text}

    코스 구성 규칙:
    - 1일 코스: 숙박은 포함하지 않아도 됨.
    - 2일 이상: 숙박은 하루 일정에 넣지 말고, 별도로 stays 리스트로만 제공.
    - 매일 관광지(POI)는 2~3곳, 음식점(MEAL)은 점심/저녁으로 1~2곳. 가까운 동선으로 묶기.
    - must_include는 반드시 최소 1회 이상 포함. exclude는 포함하지 않기.
    - 실내 우선이면 실내 위주로 배치(전시/박물관/쇼핑몰/카페 등).
    - 예산이 있으면 숙박 후보에서 1박 총액이 예산 이내인 곳을 우선 표기(정보 없으면 "예산 확인 필요" 표기).
    - 각 항목은 title, desc, reason, image(가능 시), coords(mapx,mapy가 있으면 사용/없으면 null).
    - 최종 JSON 키: keywords(5~10개), days(일자별 segments 배열), stays(숙박 후보 배열), summary(3~4문장)
    - days의 segment.type은 "POI" 또는 "MEAL".
    - 한국어로.
    """

    out = llm.invoke(prompt)
    plan = robust_json(getattr(out, "content", out), {"keywords": [], "days": [], "stays": [], "summary": ""})

    # 숙박 비어있으면 간단 보완
    if not plan.get("stays") and state.get("stays"):
        simple_stays = []
        for s in state["stays"][:5]:
            price_note = ""
            if budget:
                price_note = " (예산 확인 필요)"
            simple_stays.append({
                "title": s.get("title"),
                "desc": (s.get("desc") or "주변 숙박 후보") + price_note,
                "reason": s.get("reason") or "일정 지역과 접근성이 좋아 추천",
                "image": s.get("image") or s.get("firstimage") or s.get("firstimage2"),
                "coords": s.get("coords") or {"mapx": s.get("mapx"), "mapy": s.get("mapy")}
            })
        plan["stays"] = simple_stays

    # 응답 스냅샷을 히스토리에 적재(길이 제한)
    state["history"].append({"role":"assistant","content": json.dumps(plan, ensure_ascii=False)[:1500]})
    return {**state, "plan": plan}

# ---------------------------------------------------------------------
# 그래프 구성 (★ IncorporateFollowup 추가)
# ---------------------------------------------------------------------
graph = StateGraph(TripState)
graph.add_node("ParseQuery", node_parse_query)
graph.add_node("IncorporateFollowup", node_incorporate_followup)   # ★ 추가
graph.add_node("ResolveArea", node_resolve_area)
graph.add_node("FetchPOIs", node_fetch_pois)
graph.add_node("BuildItinerary", node_build_itinerary)

graph.set_entry_point("ParseQuery")
graph.add_edge("ParseQuery", "IncorporateFollowup")   # ★ 연결
graph.add_edge("IncorporateFollowup", "ResolveArea")
graph.add_edge("ResolveArea", "FetchPOIs")
graph.add_edge("FetchPOIs", "BuildItinerary")
graph.add_edge("BuildItinerary", END)
app = graph.compile()

# ---------------------------------------------------------------------
# 실행 예시 (대화형)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    state: TripState = {
        "userQuery": "마포에서 신나게 노는 테마로 3박 4일 여행코스 추천해줘",
        "origin": {"mapX": 126.98375, "mapY": 37.563446},  # 초기값(명동) — 지역 해석 실패 시 그대로 유지
        "days": 3,
        "mode": "walk",
        "history": [],
        "prefs": {},
        "must_include": [],
        "exclude": [],
        "catalog_cache": {}
    }

    # 1턴
    out = app.invoke(state)
    print("\n=== 1턴 결과 ===")
    print(json.dumps(out.get("plan", {}), ensure_ascii=False, indent=2))

    # 2턴(후속 지시): "홍대는 꼭 넣고, 롯데월드는 빼줘. 실내 위주로 2일로 줄여줘."
    out["userQuery"] = "홍대는 꼭 넣고, 롯데월드는 빼줘. 실내 위주로 2일로 줄여줘."
    out2 = app.invoke(out)
    print("\n=== 2턴 결과(수정 반영) ===")
    print(json.dumps(out2.get("plan", {}), ensure_ascii=False, indent=2))

    # 3턴(지역 변경): "연남동 중심으로 바꿔줘. 카페 전시 추가."
    out2["userQuery"] = "연남동 중심으로 바꿔줘. 카페 전시 추가."
    out3 = app.invoke(out2)
    print("\n=== 3턴 결과(지역/태그 반영) ===")
    print(json.dumps(out3.get("plan", {}), ensure_ascii=False, indent=2))
