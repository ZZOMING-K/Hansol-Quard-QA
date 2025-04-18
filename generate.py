from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

def generate_response(pdf_docs , csv_docs, question) : 

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.2,
        google_api_key=os.environ.get('API_KEY')
    )

    initial_prompt = ChatPromptTemplate.from_messages([
        ('system', ("""
        당신은 건설 사고 분석 전문가입니다.
        아래 세 가지 단계에 따라 사고 재발 방지 대책을 생성하세요:
        
        1. 사고 질문의 상황을 간단히 요약합니다.
        2. 유사한 사례 또는 지침에서 핵심 위험 요소를 찾아냅니다.
        3. 두 가지 핵심 대책을 직접적인 조치 문장으로 제시하세요.
        
        [참고자료]
        - 사고 관련 안전지침: {pdf_docs}
        - 유사 사고 사례 및 대응: {csv_docs}
        
        [작성 규칙]
        - 각 문장은 반드시 직접적 조치 문장 ("~를 해야 합니다") 형태여야 함
        - 서론/배경설명 생략, 중복 문장 제거
        - 각 문장의 출처(지침/사례) 인용
        """),

        ("human", "사고 질문: {question}")
    ])

    # LLM 체인 구성
    initial_chain = initial_prompt | llm | StrOutputParser()

    # 1차 응답 생성 (초안)
    initial_response = initial_chain.invoke({
        "pdf_docs": pdf_docs,
        "csv_docs": csv_docs,
        "question": question
    })

    # 2차 Self-Refine 프롬프트
    refine_prompt = ChatPromptTemplate.from_messages([
        ('system', """
        당신은 AI 초안 검토 시스템입니다.
        다음 기준에 따라 초안을 수정하세요:
        
        - 문장은 2~3문장 이내로 유지
        - 중복된 표현 제거
        - 구체적이고 간결한 조치로 변환
        - 출처는 명확하게 유지
        
        [초안]
        {initial_response}
        """)
    ])

    refine_chain = refine_prompt | llm | StrOutputParser()

    # 최종 응답 (Self-refined)
    final_response = refine_chain.invoke({
        "initial_response": initial_response
    })

    return final_response



