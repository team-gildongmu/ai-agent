# API KEY를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor, Tool
from langchain.prompts import ChatPromptTemplate
from googleapiclient.discovery import build

# API KEY 정보로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")

# ChatGPT 설정
llm = ChatOpenAI(
    model="gpt-4o",
    openai_api_key=OPENAI_API_KEY
)

# 구글 검색 API 함수 정의
def google_search(query, num_results=3):
    service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
    result = service.cse().list(q=query, cx=SEARCH_ENGINE_ID, num=num_results, lr="lang_ko").execute()

    snippets = [f"{item['title']}: {item['snippet']}" for item in result.get('items', [])]
    return "\n".join(snippets)

# LangChain 도구 설정
google_tool = Tool(
    name="GoogleSearch",
    func=google_search,
    description="여행지 구글 검색"
)

# 사용자 입력 기반 템플릿 작성
custom_prompt = ChatPromptTemplate.from_template("""
당신은 힐링 여행지를 추천하는 도우미입니다.
사용자의 연령, 성별, 여행 지역, 여행 일수, 여행 타입 정보를 바탕으로 국내 여행 코스를 추천하세요.

사용자 입력: {input}

아래는 사용자 정보입니다:
- 나이: {age}
- 성별: {gender}
- 여행 지역: {location}
- 여행 기간: {days}일
- 여행 타입: {travel_type} (예: 조용한 자연, 도심 감성, 밤문화, 역사 탐방 등)

도구 이름 리스트:
{tool_names}
                                                 
사용 가능한 도구 목록:
{tools}

에이전트가 수행한 이전 작업들:
{agent_scratchpad}

각 여행지에 대해:
1. 장소명과 간단한 설명
2. 혼잡도 정보 (예상 기준)
3. 추천 코스 (2~3시간 기준 루트)
4. 관련 꿀팁

결과는 보기 좋게 정리된 텍스트 형식으로 출력해주세요.
""")

# REACT 에이전트 생성
react_agent = create_react_agent(llm=llm, tools=[google_tool], prompt=custom_prompt)
agent_executor = AgentExecutor(agent=react_agent, tools=[google_tool], verbose=True)

# 여행지 추천 함수 정의
def recommend_travel_places(age, gender, location, days, travel_type):
    user_input = f"""
    나이: {age}
    성별: {gender}
    여행 지역: {location}
    여행 기간: {days}일
    여행 타입: {travel_type}
    """
    response = agent_executor.invoke({"input": user_input})

    print("\n🚩 추천 국내 여행 코스 🚩")
    print(response["output"])

# 실행 예시
if __name__ == "__main__":
    # 사용자 정보 입력
    age = 28
    gender = "여성"
    location = "서울"
    days = 3

    # 여행 타입 지정
    travel_type = "조용한 자연"

    recommend_travel_places(age, gender, location, days, travel_type)