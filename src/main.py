# API KEY를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType
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

# Agent 초기화
agent = initialize_agent(
    [google_tool],
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose = True #운영시 False
)

# 여행지 추천 함수 정의
def recommend_travel_places(age, gender, location, days):
    query = f"{location}에서 {age}세 {gender}이 {days}일 동안 여행하기 좋은 코스를 추천해주세요."
    response = agent.run(query)

    print("\n🚩 추천 국내 여행 코스 🚩")
    print(response)

# 실행 예시
if __name__ == "__main__":
    # 사용자 정보 입력
    age = 28
    gender = "여성"
    location = "서울"
    days = 3

    recommend_travel_places(age, gender, location, days)