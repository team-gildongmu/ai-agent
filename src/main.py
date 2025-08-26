import os
import httpx
from typing import Optional, List, Literal, Dict
from collections import defaultdict
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor

from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# =========================================================
# 환경변수 로드
# =========================================================
load_dotenv()
TOUR_KEY         = os.getenv("TOURAPI_KEY")       # 공공데이터포털 TourAPI 키
GOOGLE_API_KEY   = os.getenv("GOOGLE_API_KEY")    # Google CSE 키
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")  # CSE 엔진 ID

# =========================================================
# 공통 설정 (KorService2)
# - 공식 파라미터명 대소문자 그대로 사용
# =========================================================
TOUR_BASE = "https://apis.data.go.kr/B551011/KorService2"
COMMON_QS = {
    "serviceKey": TOUR_KEY,
    "MobileOS": "ETC",
    "MobileApp": "Gildongmu",
    "_type": "json",
}

def _get(url: str, params: Dict):
    """TourAPI 공통 GET 헬퍼"""
    merged = {**COMMON_QS, **{k: v for k, v in params.items() if v not in (None, "")}}
    with httpx.Client(timeout=20.0) as client:
        r = client.get(url, params=merged)
        r.raise_for_status()
        return r.json()

def _get_body(payload: dict) -> dict:
    """
    KorService2 표준: {header, body}
    과거 구조 호환: {response: {body}}
    """
    if not payload:
        return {}
    if isinstance(payload.get("body"), dict):
        return payload["body"]
    resp = payload.get("response")
    if isinstance(resp, dict) and isinstance(resp.get("body"), dict):
        return resp["body"]
    return {}

def _pick_items(payload: dict) -> List[dict]:
    """
    body.items.item을 안전하게 추출
    - item: list → 그대로
    - item: dict → [dict]
    - 없음/비정상 → []
    """
    body = _get_body(payload)
    items = body.get("items", None)
    if not items:
        return []
    if isinstance(items, dict):
        item = items.get("item", None)
        if item is None:
            return []
        if isinstance(item, list):
            return item
        if isinstance(item, dict):
            return [item]
        return []
    if isinstance(items, list):
        return items
    return []

def _dedup_by_contentid(items: List[dict]) -> List[dict]:
    seen = set()
    out = []
    for it in items:
        cid = str(it.get("contentid", "")).strip()
        if not cid or cid in seen:
            continue
        seen.add(cid)
        out.append(it)
    return out

def _simplify_place(it: dict) -> dict:
    """프론트/플래너 사용을 위한 최소 스키마 정규화"""
    return {
        "contentId": str(it.get("contentid", "")),
        "contentTypeId": int(it["contenttypeid"]) if it.get("contenttypeid") else None,
        "title": it.get("title"),
        "addr": (it.get("addr1") or "") + (" " + it.get("addr2") if it.get("addr2") else ""),
        "tel": it.get("tel"),
        "mapx": float(it["mapx"]) if it.get("mapx") else None,
        "mapy": float(it["mapy"]) if it.get("mapy") else None,
        "firstimage": it.get("firstimage"),
        "firstimage2": it.get("firstimage2"),
    }

# =========================================================
# TourAPI: 공식 파라미터명으로 Args 모델 정의 + 도구
# (args.model_dump(exclude_none=True) 사용)
# =========================================================

# --- 법정동 코드 조회 ldongCode2 ---
class LdongArgs(BaseModel):
    lDongRegnCd: Optional[str] = Field(None, description="법정동 시도코드 (없으면 전체 시도목록)")
    lDongListYn: Literal["N", "Y"] = Field("N", description="목록조회 여부 (N:코드조회, Y:전체목록)")

@tool("tour_ldongCode2", args_schema=LdongArgs)
def tour_ldongCode2(args: LdongArgs):
    """법정동 코드 조회(시도/시군구)"""
    return _get(f"{TOUR_BASE}/ldongCode2", args.model_dump(exclude_none=True))

# --- 분류체계 코드 조회 lclsSystmCode2 ---
class LclsArgs(BaseModel):
    lclsSystm1: Optional[str] = None
    lclsSystm2: Optional[str] = None
    lclsSystm3: Optional[str] = None
    lclsSystmListYn: Literal["N", "Y"] = Field("N", description="목록조회 여부")

@tool("tour_lclsSystmCode2", args_schema=LclsArgs)
def tour_lclsSystmCode2(args: LclsArgs):
    """분류체계 코드 조회(대/중/소)"""
    return _get(f"{TOUR_BASE}/lclsSystmCode2", args.model_dump(exclude_none=True))

# --- 위치기반 목록 locationBasedList2 ---
class LocBasedArgs(BaseModel):
    mapX: float
    mapY: float
    radius: int = Field(1000, ge=1, le=20000, description="반경(m), 최대 20000")
    contentTypeId: Optional[int] = Field(None, description="12/14/15/25/28/32/38/39")
    arrange: Literal["A","C","D","E","O","Q","R","S"] = Field("E", description="정렬")
    numOfRows: int = 50
    pageNo: int = 1

@tool("tour_locationBasedList2", args_schema=LocBasedArgs)
def tour_locationBasedList2(args: LocBasedArgs):
    """위치기반 목록 조회"""
    return _get(f"{TOUR_BASE}/locationBasedList2", args.model_dump(exclude_none=True))

# --- 지역기반 목록 areaBasedList2 ---
class AreaBasedArgs(BaseModel):
    areaCode: Optional[str] = None
    sigunguCode: Optional[str] = None
    lDongRegnCd: Optional[str] = None
    lDongSignguCd: Optional[str] = None
    contentTypeId: Optional[int] = None
    arrange: Literal["A","C","D","O","Q","R"] = "C"
    numOfRows: int = 50
    pageNo: int = 1

@tool("tour_areaBasedList2", args_schema=AreaBasedArgs)
def tour_areaBasedList2(args: AreaBasedArgs):
    """지역/시군구 또는 법정동 기반 목록"""
    return _get(f"{TOUR_BASE}/areaBasedList2", args.model_dump(exclude_none=True))

# --- 키워드 검색 searchKeyword2 ---
class KeywordArgs(BaseModel):
    keyword: str
    contentTypeId: Optional[int] = None
    areaCode: Optional[str] = None
    sigunguCode: Optional[str] = None
    numOfRows: int = 30
    pageNo: int = 1
    arrange: Literal["A","C","D","O","Q","R"] = "C"

@tool("tour_searchKeyword2", args_schema=KeywordArgs)
def tour_searchKeyword2(args: KeywordArgs):
    """키워드 검색"""
    return _get(f"{TOUR_BASE}/searchKeyword2", args.model_dump(exclude_none=True))

# --- 축제/행사 searchFestival2 ---
class FestivalArgs(BaseModel):
    eventStartDate: str  # YYYYMMDD (필수)
    eventEndDate: Optional[str] = None
    areaCode: Optional[str] = None
    sigunguCode: Optional[str] = None
    numOfRows: int = 30
    pageNo: int = 1
    arrange: Literal["A","C","D","O","Q","R"] = "C"

@tool("tour_searchFestival2", args_schema=FestivalArgs)
def tour_searchFestival2(args: FestivalArgs):
    """행사/공연/축제 기간 조회"""
    return _get(f"{TOUR_BASE}/searchFestival2", args.model_dump(exclude_none=True))

# --- 숙박 searchStay2 ---
class StayArgs(BaseModel):
    areaCode: Optional[str] = None
    sigunguCode: Optional[str] = None
    lDongRegnCd: Optional[str] = None
    lDongSignguCd: Optional[str] = None
    lclsSystm1: Optional[str] = None
    lclsSystm2: Optional[str] = None
    lclsSystm3: Optional[str] = None
    numOfRows: int = 30
    pageNo: int = 1
    arrange: Literal["A","C","D","O","Q","R"] = "C"

@tool("tour_searchStay2", args_schema=StayArgs)
def tour_searchStay2(args: StayArgs):
    """숙박 정보 조회(법정동/분류체계 활용 가능)"""
    return _get(f"{TOUR_BASE}/searchStay2", args.model_dump(exclude_none=True))

# --- 상세 공통 detailCommon2 ---
class DetailCommonArgs(BaseModel):
    contentId: str

@tool("tour_detailCommon2", args_schema=DetailCommonArgs)
def tour_detailCommon2(args: DetailCommonArgs):
    """공통 상세(개요/좌표/주소/분류/대표이미지 등)"""
    return _get(f"{TOUR_BASE}/detailCommon2", args.model_dump(exclude_none=True))

# --- 상세 소개(타입별) detailIntro2 ---
class DetailIntroArgs(BaseModel):
    contentId: str
    contentTypeId: int

@tool("tour_detailIntro2", args_schema=DetailIntroArgs)
def tour_detailIntro2(args: DetailIntroArgs):
    """타입별 소개 상세(운영/이용/편의/환불 등)"""
    return _get(f"{TOUR_BASE}/detailIntro2", args.model_dump(exclude_none=True))

# --- 상세 반복정보 detailInfo2 ---
class DetailInfoArgs(BaseModel):
    contentId: str
    contentTypeId: int

@tool("tour_detailInfo2", args_schema=DetailInfoArgs)
def tour_detailInfo2(args: DetailInfoArgs):
    """반복 상세(객실/코스 등)"""
    return _get(f"{TOUR_BASE}/detailInfo2", args.model_dump(exclude_none=True))

# --- 상세 이미지 detailImage2 ---
class DetailImageArgs(BaseModel):
    contentId: str
    imageYN: Literal["Y","N"] = "Y"
    numOfRows: int = 10
    pageNo: int = 1

@tool("tour_detailImage2", args_schema=DetailImageArgs)
def tour_detailImage2(args: DetailImageArgs):
    """이미지 상세(저작권 유형 포함 cpyrhtDivCd)"""
    return _get(f"{TOUR_BASE}/detailImage2", args.model_dump(exclude_none=True))

# =========================================================
# Google CSE (보강/검증)
# =========================================================
from googleapiclient.discovery import build

class GArgs(BaseModel):
    query: str
    num_results: int = 3
    lr: Literal["lang_ko","lang_en"] = "lang_ko"

@tool("google_search", args_schema=GArgs)
def google_search(args: GArgs):
    """운영시간/휴무/리뷰/공지 등 보강용 검색"""
    service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
    res = service.cse().list(
        q=args.query,
        cx=SEARCH_ENGINE_ID,
        num=args.num_results,
        lr=args.lr
    ).execute()
    return res.get("items", [])

# =========================================================
# Wrapper: 자연어 카테고리 → contentTypeId 매핑 + 검색/정제
# =========================================================
CONTENT_TYPES: Dict[str, int] = {
    # 핵심 매핑
    "자연": 12,  # 사용자/LLM이 "자연"이라고 말해도 관광지(12)로 매핑
    "관광지": 12,
    "문화시설": 14,
    "축제": 15, "공연": 15, "행사": 15,
    "여행코스": 25,
    "레포츠": 28,
    "숙박": 32, "호텔": 32, "펜션": 32, "게스트하우스": 32,
    "쇼핑": 38,
    "음식점": 39, "맛집": 39, "카페": 39,
}

class DiscoverArgs(BaseModel):
    """
    여행 후보 탐색 래퍼.
    - categories: ["자연","숙박","음식점"] 등 자연어 카테고리(없으면 기간 기준 기본값)
    - days: 1(당일) or N(숙박 포함)
    - 좌표 or 지역코드를 최소 하나 제공
    - keyword가 있으면 키워드 검색 우선 사용(타입별 필터)
    """
    categories: Optional[List[str]] = Field(None, description="예: 자연/관광지/숙박/음식점/축제/쇼핑 등(자연어)")
    days: int = Field(1, ge=1, description="여행 일수. 1 초과면 자동으로 숙박 포함")
    # 좌표 기반
    mapX: Optional[float] = None
    mapY: Optional[float] = None
    radius: int = Field(3000, ge=1, le=20000, description="위치기반 반경(m), 최대 20000")
    # 지역 코드 기반
    areaCode: Optional[str] = None
    sigunguCode: Optional[str] = None
    lDongRegnCd: Optional[str] = None
    lDongSignguCd: Optional[str] = None
    # 키워드
    keyword: Optional[str] = Field(None, description="키워드가 있으면 검색 우선 적용")
    # 공통
    per_type_limit: int = Field(20, ge=1, le=100, description="타입별 최대 가져올 개수")

@tool("poi_discover", args_schema=DiscoverArgs)
def poi_discover(args: DiscoverArgs) -> dict:
    """
    자연어 카테고리를 contentTypeId로 변환해 TourAPI를 여러 번 호출하고, 결과를 합쳐 정규화하여 반환.
    분기:
      - keyword가 있으면 searchKeyword2 사용(타입별 필터)
      - 아니면 좌표 있으면 locationBasedList2, 없으면 areaBasedList2 (또는 법정동 코드 활용)
    """
    # 1) 카테고리 결정
    cats = list(args.categories or [])
    if not cats:
        cats = ["관광지", "음식점"] if args.days <= 1 else ["관광지", "음식점", "숙박"]

    type_ids: List[int] = []
    for c in cats:
        t = CONTENT_TYPES.get(c)
        if t and t not in type_ids:
            type_ids.append(t)
    if not type_ids:
        type_ids = [12, 39] if args.days <= 1 else [12, 39, 32]

    batches: Dict[int, List[dict]] = defaultdict(list)

    # 2) 호출 분기: 키워드 → 좌표 → 지역
    if args.keyword:
        for t in type_ids:
            payload = _get(f"{TOUR_BASE}/searchKeyword2", {
                **KeywordArgs(
                    keyword=args.keyword,
                    contentTypeId=t,
                    areaCode=args.areaCode,
                    sigunguCode=args.sigunguCode,
                    numOfRows=args.per_type_limit,
                    pageNo=1,
                    arrange="C"
                ).model_dump(exclude_none=True)
            })
            items = _pick_items(payload)
            batches[t].extend(items)
    else:
        if args.mapX is not None and args.mapY is not None:
            for t in type_ids:
                payload = _get(f"{TOUR_BASE}/locationBasedList2", {
                    **LocBasedArgs(
                        mapX=args.mapX, mapY=args.mapY,
                        radius=args.radius, contentTypeId=t,
                        numOfRows=args.per_type_limit, pageNo=1, arrange="E"
                    ).model_dump(exclude_none=True)
                })
                items = _pick_items(payload)
                batches[t].extend(items)
        else:
            for t in type_ids:
                payload = _get(f"{TOUR_BASE}/areaBasedList2", {
                    **AreaBasedArgs(
                        areaCode=args.areaCode,
                        sigunguCode=args.sigunguCode,
                        lDongRegnCd=args.lDongRegnCd,
                        lDongSignguCd=args.lDongSignguCd,
                        contentTypeId=t, numOfRows=args.per_type_limit, pageNo=1, arrange="C"
                    ).model_dump(exclude_none=True)
                })
                items = _pick_items(payload)
                batches[t].extend(items)

    # 3) 타입별 중복제거 후 합치기
    merged: List[dict] = []
    for t, items in batches.items():
        dedup = _dedup_by_contentid(items)[:args.per_type_limit]
        merged.extend(dedup)

    # 4) 전체 중복 제거 및 정규화
    merged = _dedup_by_contentid(merged)
    simplified = [_simplify_place(it) for it in merged]

    return {
        "requested_categories": cats,
        "used_type_ids": type_ids,
        "count": len(simplified),
        "results": simplified,
    }

# =========================================================
# Wrapper: 상세/이미지 일괄 보강
# =========================================================
class EnrichArgs(BaseModel):
    """
    후보 목록에 대해 detailCommon2 / detailIntro2 / detailImage2 를 호출해 보강.
    - entries: [{contentId, contentTypeId}]
    - with_images: 이미지 조회 여부
    """
    entries: List[dict] = Field(..., description="각 항목은 {contentId, contentTypeId}")
    with_images: bool = True
    image_rows: int = Field(5, ge=1, le=30)

@tool("poi_enrich_details", args_schema=EnrichArgs)
def poi_enrich_details(args: EnrichArgs) -> dict:
    out = []
    for e in args.entries:
        cid = str(e.get("contentId") or "").strip()
        ctype = e.get("contentTypeId")
        if not cid:
            continue

        block = {"contentId": cid, "contentTypeId": ctype}

        # detailCommon2
        try:
            common = tour_detailCommon2.invoke(DetailCommonArgs(contentId=cid))
            block["common"] = (_pick_items(common) or [{}])[0]
        except Exception as ex:
            block["common_error"] = str(ex)

        # detailIntro2 (ctype 있을 때만)
        if ctype:
            try:
                intro = tour_detailIntro2.invoke(DetailIntroArgs(contentId=cid, contentTypeId=ctype))
                block["intro"] = (_pick_items(intro) or [{}])[0]
            except Exception as ex:
                block["intro_error"] = str(ex)

        # detailImage2
        if args.with_images:
            try:
                image = tour_detailImage2.invoke(DetailImageArgs(contentId=cid, imageYN="Y", numOfRows=args.image_rows, pageNo=1))
                block["images"] = _pick_items(image)
            except Exception as ex:
                block["images_error"] = str(ex)

        out.append(block)

    return {"count": len(out), "items": out}

# =========================================================
# 오케스트레이션 에이전트 (create_tool_calling_agent + 메시지 히스토리)
# =========================================================
llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

SYSTEM_RULES = """
당신은 한국 여행 코스 플래너입니다.
규칙:
1) 장소 후보: poi_discover(자연어→contentTypeId 자동 변환)를 우선 사용. 좌표 있으면 위치기반(E:거리순, radius≤20km), 지역명이면 ldongCode2→areaBasedList2.
2) 후보 각각은 poi_enrich_details로 detailCommon/Intro/Image를 채운 뒤, 필요 시 google_search로 운영시간/휴무/공지/리뷰를 보강.
3) 여행이 1일 초과면 숙박(32)과 식당(39)을 포함해 동선 최적화 코스를 생성. 도보/차량 여부가 있으면 이동시간을 고려해 타임블록화.
4) 사용자가 기간/지역/테마를 언급하지 않으면 기본: 기간=당일, 중심좌표=프론트에서 전달한 현재 좌표.
5) 이미지는 cpyrhtDivCd를 확인해 출처 표기를 준비.
6) 답변은 요약 + 코스 표(일자·시간·이동수단·예상소요) + 팁(혼잡 회피/우천 대안).
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_RULES),
    MessagesPlaceholder("chat_history"),        # ✅ 히스토리 주입
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),    # 에이전트 내부 scratchpad
])

tools = [
    # 래퍼(상위 추상화)
    poi_discover,
    poi_enrich_details,

    # 원본 세부 도구
    tour_ldongCode2, tour_lclsSystmCode2,
    tour_locationBasedList2, tour_areaBasedList2, tour_searchKeyword2,
    tour_searchFestival2, tour_searchStay2,
    tour_detailCommon2, tour_detailIntro2, tour_detailInfo2, tour_detailImage2,

    # 보강
    google_search,
]

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
)

# =========================
# RunnableWithMessageHistory 설정
# =========================
_SESSION_STORE: dict[str, ChatMessageHistory] = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in _SESSION_STORE:
        _SESSION_STORE[session_id] = ChatMessageHistory()
    return _SESSION_STORE[session_id]

runnable = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    history_messages_key="chat_history",  # ✅ 프롬프트의 placeholder 키와 동일
    input_messages_key="input",
    output_messages_key="output",
)

# =========================================================
# 로컬 테스트 예시
# =========================================================
if __name__ == "__main__":
    # 좌표 + 1박2일 + "자연" 키워드가 섞여도 정상 (자연→12 매핑)
    user_lat, user_lon = 37.5665, 126.9780  # 서울시청 근처
    user_input = f"""
    서울에서 '자연' 위주로 1박 2일 여행 코스를 추천해줘.
    내 현재 좌표는 ({user_lat}, {user_lon})야. 도보 중심으로 이동하고 싶어.
    """

    # 세션 ID는 사용자/대화방 기준으로 설정
    config = {"configurable": {"session_id": "demo-user-001"}}

    # ✅ chat_history는 자동 관리
    res1 = runnable.invoke({"input": user_input}, config=config)
    print("\n=== 1턴 응답 ===")
    print(res1.get("output", ""))

    res2 = runnable.invoke({"input": "비 오면 실내 대안도 함께 넣어줘."}, config=config)
    print("\n=== 2턴 응답 ===")
    print(res2.get("output", ""))