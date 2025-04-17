from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.document_loaders import TextLoader, DirectoryLoader, DataFrameLoader
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from kiwipiepy import Kiwi

from langchain_text_splitters import KonlpyTextSplitter

def langchain_embeddings(config):
    '''
    랭체인 임베딩 설정 함수.
    :param config:
    :return:
    '''
    if config['embedding_type'] == 'huggingface':
        embedding = HuggingFaceEmbeddings(config['embedding_model_name']
                                          ,model_kwargs = {'device' : 'cuda'},
                                          encode_kwargs={'normalize_embeddings': True})
    elif config['embedding_type'] == 'gemini':
        embedding = GoogleGenerativeAIEmbeddings(config['embedding_model_name'])

    return embedding

def document_loader(config):
    '''
    문서 업로드 함수
    :param config:
    :return:
    '''
    if config['doc_type'] == 'markdown':
        # 마크다운 처리
        mark_path = config['data_path']
        text_loader_kwargs = {"autodetect_encoding": True}

        # markdown loader
        loader = DirectoryLoader(
            mark_path,
            glob="*.md",
            loader_cls=TextLoader,
            silent_errors=True,
            loader_kwargs=text_loader_kwargs,
        )
        docs = loader.load()

        # 잘 올라갔는지 확인

        print(docs[0].page_content)
        print(len(docs))

        # 마크다운 문서 처리
        # 헤더로 나누어져있지 않아 기존 코드 사용
        text_splitter = KonlpyTextSplitter(chunk_size=config['chunk_size'], chunk_overlap=config['chunk_overlap'])
        docs = text_splitter.split_documents(docs)

        return docs
    elif config['doc_type'] == 'qa': # qa 데이터셋 로드하는 경우
        loader = DataFrameLoader(config['dataframe'], page_content_column="context")
        datasets_docs = loader.load()
        # qa vector db 만들기
        return datasets_docs


def store_vectordb(docs, embeddings, save_path):
    # FAISS가 빨라서 사용했습니다.
    vectordb = FAISS.from_documents(docs, embeddings, distance_strategy=DistanceStrategy.COSINE)
    vectordb.save_local(save_path)

    return vectordb

def get_retriever(markdb, qadb, k, docs, dataset_docs):

    # kiwi tokenizer for bm25
    kiwi = Kiwi()

    # kiwi 토크나이징 함수(키워드 기반)
    def preprocessing_with_kiwi(text):
        return [t.form for t in kiwi.tokenize(text)]

    # pdf(markdown) retriever
    pdf_retriever = markdb.as_retriever(search_kwargs={'k': k}, search_type='similarity')

    # qa retriever
    qa_retriever = qadb.as_retriever(search_kwargs={'k': k}, search_type='similarity')

    # BM25 Retirever
    bm_pdf_retriever = BM25Retriever.from_documents(docs,
                                                    preprocess_func=preprocessing_with_kiwi)
    bm_qa_retriever = BM25Retriever.from_documents(dataset_docs,
                                                   preprocess_func=preprocessing_with_kiwi)
    bm_pdf_retriever.k = k
    bm_qa_retriever.k = k

    # ensemble retriever
    ensemble_pdf_retriever = EnsembleRetriever(
        retrievers=[pdf_retriever, bm_pdf_retriever],
        weights=[0.5, 0.5]
    )

    ensemble_qa_retriever = EnsembleRetriever(
        retrievers=[qa_retriever, bm_qa_retriever],
        weights=[0.5, 0.5]
    )

    # Reranker 사용
    encoder_model = HuggingFaceCrossEncoder(model_name='BAAI/bge-reranker-v2-m3')
    compressor = CrossEncoderReranker(model=encoder_model, top_n=3)

    final_doc_retriever = ContextualCompressionRetriever(base_compressor=compressor,
                                                         base_retriever=ensemble_pdf_retriever)
    final_qa_retriever = ContextualCompressionRetriever(base_compressor=compressor,
                                                        base_retriever=ensemble_qa_retriever)

    return final_doc_retriever, final_qa_retriever
