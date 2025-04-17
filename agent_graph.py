# state 정의 하기
from typing import List
from typing_extensions import TypedDict
from langgraph.graph import END, START, StateGraph
from langchain_core.prompts import ChatPromptTemplate ,  MessagesPlaceholder
from generate import generate_response
from langchain_core.pydantic_v1 import BaseModel, Field
from preprocessing import prepro_data
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from generate import generate_response
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from retriever import get_retriever
from generate import generate_response
import os 


pdf_retriever , csv_retriever = get_retriever(pdf_db , qa_db , k = 3 , pdf_docs , csv_docs)

# 1. 데이터 로드 
data = prepro_data("./data/prepro_train.csv")

# 2. 임베딩 모델 로드 (임베딩 파일)
embedding_model_name = "intfloat/multilingual-e5-large-instruct"

hf_embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name ,
                                        model_kwargs = {"device" : "cuda" , "trust_remote_code" : True} , # cuda, cpu
                                        encode_kwargs = {"normalize_embeddings" : True}) # set True for cosine similarity

# 3. 벡터 디비 로드 (저장된 벡터 디비 가져오기)
pdf_db = FAISS.load_local('vectordb/pdf_faiss', 
                          hf_embeddings,
                          allow_dangerous_deserialization=True)
csv_db = FAISS.load_local('vectordb/train_faiss', 
                          hf_embeddings,
                          allow_dangerous_deserialization=True)

# 4. retreiver 함수 정의 
pdf_retriever , csv_retriever = get_retriever(pdf_db , qa_db , k = 3 , pdf_dataset , csv_dataset)

# 5. generate 함수 정의 (generate 파일)

#################################################################

class GraphState(TypedDict) :
  question : str
  pdf_question : str
  generation : str
  pdf_docs : List[str]
  csv_docs : List[str]

def transform_pdf_query(state) :
    
    system = """
        당신은 건설 안전 분야 전문 쿼리 변환기입니다. 입력된 건설 사고 관련 질문을 안전보건작업지침 문서 벡터 검색에 최적화된 형태로 재작성하는 역할을 합니다.
        """

    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Here is the initial question: \n\n {question} \n Formulate an improved question.",
            ),
        ]
    )

    question_rewriter = re_write_prompt | llm | StrOutputParser()

    question = state['question']

    better_question = question_rewriter.invoke({"question": question})

    print(better_question)

    return {"pdf_question" : better_question}

def retrieve(state) :

    print("--RETRIEVE--")

    csv_question = state['question'] # origin question
    pdf_question = state['pdf_question'] # 재작성된 쿼리

    csv_docs = csv_retriever.invoke(csv_question)
    pdf_docs = pdf_retriever.invoke(pdf_question)
    
    example_prompt = []
    
    for i , doc in enumeerate(csv_docs) :
        
        context = doc.page_content
        question = doc.metadata['question']
        answer = doc.metadata['answer']
      
        example_prompt.append(f'question:{context} {question}\nanswer:{answer}\n\n')

  return {"pdf_docs" : pdf_docs , "csv_docs" : example_prompt , "question" : csv_question , "pdf_question" : pdf_question}

def generate(state) :

  print("--GENERATE--")

  question = state['question']

  pdf_docs = state['pdf_docs']
  csv_docs = state['csv_docs']

  rag_chain = generate_response(pdf_docs, csv_docs, question)
  generation = rag_chain.invoke({"pdf_docs": pdf_docs, "csv_docs" : csv_docs ,"question": question})

  return {"pdf_docs" : pdf_docs , "csv_docs" : csv_docs , "question" : question , "generation" : generation}

def grade_documents(state) :
    
    class GradeDocuments(BaseModel):
        """Binary score for relevance check on retrieved documents."""

        binary_score: str = Field(
            description="Documents are relevant to the question, 'yes' or 'no'"
        )
        
    structured_llm_grader = llm.with_structured_output(GradeDocuments)

    # Prompt
    system = """You are a grader assessing relevance of a retrieved document to a user question. \n
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved document: \n\n {pdf_docs} \n\n User question: {question}"),
        ]
    )

    retrieval_grader = grade_prompt | structured_llm_grader

    print("---CHECK PDF DOCUMENT RELEVANCE TO QUESTION---")

    pdf_question = state["pdf_question"]

    pdf_docs = state["pdf_docs"]

    filtered_pdf_docs = []

    for p_doc in pdf_docs :

        score = retrieval_grader.invoke({"question" : pdf_question , "pdf_docs" : p_doc.page_content})

        grade = score.binary_score

        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_pdf_docs.append(p_doc)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue

    if not filtered_pdf_docs:
        print("참조할 안전보건지침서가 없습니다.")

    return {"pdf_docs" : filtered_pdf_docs , "pdf_question" : pdf_question}


workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)  
workflow.add_node("grade_documents", grade_documents)  
workflow.add_node("generate", generate)  
workflow.add_node("transform_query", transform_pdf_query) 

# Build Graph
workflow.add_edge(START, "transform_query") 
workflow.add_edge("transform_query" , "retrieve") 
workflow.add_edge("retrieve" , "grade_documents") 
workflow.add_edge("grade_documents" , "generate") 
workflow.add_edge("generate" , END)

# Compile
graph = workflow.compile()