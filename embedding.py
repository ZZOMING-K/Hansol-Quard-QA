from typing import List, Tuple
import os
import glob
import re
import pickle
import faiss
from langchain.docstore.document import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain.document_loaders import DataFrameLoader

def load_markdown_files(folder_path: str) -> List[Tuple[str, str]]:
    """
    마크 다운 파일들을 로드하여(파일명, 텍스트) 리스트로 반환하는 함수
    """

    file_paths = glob.glob(os.path.join(folder_path, "*.md"))

    return [(path, open(path, "r", encoding="utf-8").read()) for path in file_paths]


def custom_md_splitter(md_text: str) -> List[Document]:
    """
    마크다운 헤더가 현재 md 파일에 없기 때문에 챕터 구분 별로 (예 : 1. 목적 2. 배경)
    나누기 위한 함수.
    """
    pattern = re.compile(r"(?P<header>^\d+(\.\d+)*\s+.+)", re.MULTILINE)
    matches = list(pattern.finditer(md_text))

    documents = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md_text)

        header = match.group("header").strip()
        content = md_text[start:end].strip()

        # 컨텐츠에는 헤더로 분리한 것도 내용과 합치고, 메타데이터에서 "section" 으로 구분
        
        if content:
            doc = Document(
                page_content=f"{header}\n{content}", metadata={"section": header}
            )
            documents.append(doc)

    return documents


def semantic_chunk_documents(
    documents: List[Document], model: SentenceTransformer) -> List[str]:
    
    """
    텍스트를 의미적 유사도 기준으로 청킹하는 함수.
    """
    
    chunker = SemanticChunker(embeddings=model)
    
    final_chunks = []
    
    for doc in documents:
        chunks = chunker.split_text(doc.page_content)
        final_chunks.extend(chunks)
    
    return final_chunks


def load_existing_faiss(output_dir: str):
    """
    기존 저장된 벡터DB 가져오는 함수
    """
    faiss_path = os.path.join(output_dir, "faiss.index")
    meta_path = os.path.join(output_dir, "metadata.pkl")

    if os.path.exists(faiss_path) and os.path.exists(meta_path):
        index = faiss.read_index(faiss_path)
        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)
        # 기존 파일 경로 추출
        existing_paths = set([line.split(" | ")[0] for line in metadata])
        return index, metadata, existing_paths
    else:
        return None, [], set()


def save_faiss(index, metadata: List[str], output_dir: str):
    """
    벡터DB 저장 함수
    """
    os.makedirs(output_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(output_dir, "faiss.index"))
    with open(os.path.join(output_dir, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)
    print(f"✅ 저장 완료: {output_dir}")


def process_md_folder(
    md_folder_path: str,
    output_dir: str,
    model_name="intfloat/multilingual-e5-large",
):
    """
    최종적으로 벡터DB를 만드는 함수. 입력한 경로에 기존 벡터DB가 없다면 새로 생성성

    # 파라미터
    - md_folder_path : md 파일이 모여있는 폴더 이름 입력
    - output_dir : 기존 저장되어 잇는 벡터 (md : vectordb/faiss_md_index)
    """
    model = SentenceTransformer(model_name)
    md_files = load_markdown_files(md_folder_path)

    # 기존 인덱스, 메타데이터, 파일경로 목록 로드
    index, metadata, existing_paths = load_existing_faiss(output_dir)
    if index is None:
        print("🔄 새로운 FAISS 인덱스 생성 중...")
        index = faiss.IndexFlatL2(model.get_sentence_embedding_dimension())
        metadata = []
        existing_paths = set()

    new_chunks = []
    new_metadata = []

    for path, text in md_files:
        if path in existing_paths:
            print(f"⏭ 이미 처리됨: {path}")
            continue

        docs = custom_md_splitter(text)
        chunks = semantic_chunk_documents(docs, model)
        print(f"📄 {os.path.basename(path)} → {len(chunks)} chunks")

        new_chunks.extend(chunks)
        new_metadata.extend([f"{path} | {chunk[:50]}" for chunk in chunks])

    if new_chunks:
        embeddings = model.encode(new_chunks, show_progress_bar=True)
        index.add(embeddings)
        metadata.extend(new_metadata)
        save_faiss(index, metadata, output_dir)
    else:
        print("✅ 추가할 새로운 문서가 없습니다.")
        

#----------------------------------------------------------------------------------# 

def split_documents(chunk_size, KB):

  text_splitter = RecursiveCharacterTextSplitter(
        separators=[ "\n\n", "\n", ".", " ", ""] ,
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 5),
        length_function = len ,
        is_separator_regex=True
    )

  # 문서 리스트를 순회하면서 청크 생성

  docs_processed = []

  for doc in KB:
      docs_processed += text_splitter.split_documents([doc])

  return docs_processed

def load_pdf_docs(pdf_path = './pdf2txt/') : 
    

    text_loader_kwargs = {"autodetect_encoding": True}

    loader = DirectoryLoader(
        pdf_path,
        glob="*.md",
        loader_cls=TextLoader,
        silent_errors=True,
        loader_kwargs=text_loader_kwargs,
    )
    
    docs = loader.load()
    
    docs_processed= split_documents(chunk_size = 500, KB = docs)
    
    return docs_processed

def load_csv_docs(data) : 
    
    combined_csv_data = data.apply(
    
    lambda row: {
        "context": (
            f"'{row['공종']}' 중 {row['사고원인']}'로 인해 사고가 발생했습니다. "
            f"해당 사고는 '{row['작업프로세스']}' 중 발생했으며, 관련 사고객체는 '{row['부위']}'입니다. "
            f"이로 인한 인적피해는 '{row['인적사고']}' 이고, 물적피해는 '{row['물적사고']}'로 확인됩니다."
        ),
        "question" : f"해당 사고의 재발 방지 대책과 향후 조치 계획은 무엇인가요?",
        "answer": row["재발방지대책 및 향후조치계획"]
    },
    axis=1
    )
    
    loader = DataFrameLoader(combined_csv_data, page_content_column="context")
    datasets_docs = loader.load()
    
    return dataset_docs

#  embedding model load 
def load_embedding_model(model_name = "intfloat/multilingual-e5-large") :

    embedding_model_name = model_name

    hf_embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name ,
                                        model_kwargs = {"device" : "cuda" , "trust_remote_code" : True} , # cuda, cpu
                                        encode_kwargs = {"normalize_embeddings" : True}) # set True for cosine similarity
    
    return hf_embeddings

# vector db load 
def load_vector_db(embedding_model) : 
    
    pdf_db = FAISS.load_local('./vectordb/pdf_faiss' , 
                              embedding_model,
                              allow_dangerous_deserialization = True
                              )

    csv_db = FAISS.load_local('./vectordb/csv_faiss' , 
                              embedding_model ,
                              allow_dangerous_deserialization = True 
                              )
    
    return pdf_db , csv_db
