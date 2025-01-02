import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_upstage import UpstageDocumentParseLoader
import re

from pinecone import Pinecone

# from langchain_chroma import Chroma
from langchain_upstage import UpstageEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_pinecone import PineconeVectorStore

import openai
# from langchain.prompts import ChatPromptTemplate
# from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
# from langchain.schema import SystemMessage, HumanMessage

# warnings.filterwarnings("ignore")
def clean_parsed_text(text):
    # 이스케이프 시퀀스 제거
    text = text.replace('\\"', '"').replace('\\n', '\n')
    
    # 불필요한 공백 제거
    text = re.sub(r'\s+', ' ', text)
    
    # HTML 태그 제거 (선택적)
    text = re.sub(r'<[^>]+>', '', text)
    
    return text.strip()

# layzer.lazy_load()
# 메모리 효율을 향상시키기 위해서, lazy_load() 로 페이지별로 문서를 불러올 수도 있음

#### Text Split

def loadDocs() :
    # .env 파일 로드
    # dotenv_path = os.path.join(os.path.dirname(__file__), './env')
    # print(dotenv_path)
    # load_dotenv(dotenv_path)
    load_dotenv()
    #### SET API KEY

    # Upstae API KEY
    os.environ["UPSTAGE_API_KEY"] = os.getenv("UPSTAGE_API_KEY")
    # Pinecone API KEY
    os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    #### Document Loading


    layzer = UpstageDocumentParseLoader(
        "sqld_summary.pdf", # 불러올 파일
        output_format='html',  # 결과물 형태 : HTML
        coordinates= False) # 이미지 OCR 좌표계 가지고 오지 않기

    docs = layzer.load()
    return docs


def getSplit(docs) :
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100)

    splits = text_splitter.split_documents(docs)

    for document in splits :
      document.page_content = clean_parsed_text(document.page_content)

    # print("Splits:", len(splits))
    return splits


#### VECTOR STORE
def getVectorStoreFromExisting():
    # Pinecone API 초기화
    api_key = os.getenv("PINECONE_API_KEY")
    print(api_key)
    # environment = "us-east-1"  # Pinecone 환경 (콘솔에서 확인 가능)
    pc = Pinecone(api_key=api_key)

    # 기존 인덱스 이름
    index_name = "sqld-exam"

    # 기존 인덱스 불러오기
    if index_name not in pc.list_indexes().names():
        raise ValueError(f"Index '{index_name}' does not exist in Pinecone.")

    # PineconeVectorStore 객체 생성
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=UpstageEmbeddings(model="embedding-query")
    )

    return vectorstore

# def getVectorStore(splits) :
#     # 3. Embed & indexing
#     index_name = "sqld-exam"
#     pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

#     # create new index
#     if index_name not in pc.list_indexes().names():
#         pc.create_index(
#             name=index_name,
#             dimension=4096,
#             metric="cosine",
#             spec=ServerlessSpec(cloud="aws", region="us-east-1"),
#         )
#     vectorstore = PineconeVectorStore.from_documents(
#         splits, UpstageEmbeddings(model="embedding-query"), index_name="sqld-exam"
#     )
#     return vectorstore


def getRetriever(vectorstore) :
    retriever = vectorstore.as_retriever(
    search_type= 'mmr', # default : similarity(유사도) / mmr 알고리즘
    search_kwargs={"k": 3} # 쿼리와 관련된 chunk를 3개 검색하기 (default : 4)
    )
    return retriever

#### 연습문제 출력하기위해선 Prompt 수정 필요요
def getChain() :
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    #### Creating a Prompt with Retrieved Result
    # OpenAI Chat Model 설정 (gpt-4 또는 gpt-4o-mini 사용 가능)
    llm = ChatOpenAI(
        model="gpt-4o-mini",  # 또는 "gpt-4o-mini"
        temperature=0.7,  # 응답의 창의성 조정
        max_tokens=1500   # 최대 응답 토큰 수
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are an AI assistant designed for question-answering tasks.
                Use the provided context and conversation history to answer the user's question.
                If you don't know the answer, simply state that you don't have the information.

                If question is about SQLD exam, please complete the following three tasks
                1. Provide a detailed explanation of the concept addressed in the user's question.
                2. Generate between three and five multiple-choice questions related to the concept. 
                    Each question should have four options (A, B, C, D).
                3. List the correct answers for all multiple-choice questions at the end of your response.

                Remember to use the given context to inform your answers and ensure accuracy in your explanations and questions. If the retriever doesn't provide relevant information, don't attempt to answer the question and instead use the specified response.
                ---
                CONTEXT:
                {context}
                """,
            ),
            ("human", "{input}"),
        ]
    )


    #### Implementing an LLM Chain
    chain = prompt | llm | StrOutputParser()
    return chain


