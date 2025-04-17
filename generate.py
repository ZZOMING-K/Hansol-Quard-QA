from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

def generate_response(pdf_docs , csv_docs, question) : 

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        google_api_key=os.environ.get('API_KEY')
    )

    prompt = ChatPromptTemplate.from_messages([
        ('system', (
            '당신은 건설 현장에서 발생한 안전사고를 분석하고 재발 방지 대책 및 향후 조치 계획을 제시하는 AI 도우미입니다. '
            '다음에 주어지는 "사고 관련 안전지침"과 "유사한 사고사례 및 대응책"을 참고하여 답변을 작성하세요 '
            '### 1. 사고 관련 안전지침 '
            '아래는 건설 현장에서 사고를 방지하기 위한 안전 지침입니다.'
            '{pdf_docs}'
            '### 2. 유사한 사고 사례 및 대응책'
            '아래는 유사한 사고 사례와 그에 대한 재발 방지 대책 및 향후 조치 계획입니다. '
            '{csv_docs}'
            '서론, 배경설명, 추가설명 없이 대응책을 핵심 내용만 요약해서 간략하게 작성하세요. '
            '반드시 한국어로 답변하며, 주어진 사고 상황에 따른 재발 방지 대책 및 향후 조치 계획을 제시하세요. '
            '제공한 자료들 중에 어떤 부분을 인용했는지 그리고 어떻게 추론해서 답변을 도출했는지 자세하게 작성해주세요.'
        )),

        ('human', '{question}')
    ])

    # 체인 구성
    rag_chain = prompt | llm | StrOutputParser()
    
    return rag_chain