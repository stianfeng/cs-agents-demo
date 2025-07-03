import logging
from typing import List, Dict, Literal
import joblib
import bs4
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

import chromadb
from chromadb.api import ClientAPI

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.utils.function_calling import tool_example_to_messages
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_core.tools import tool

from settings import settings
from recommender import MobilePlanRecommender
from schema import MobilePlanRequest, MobilePlan

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize the recommender
recommender = MobilePlanRecommender('./data/mobile.csv')
recommender.load('./data')

# Initialize the LLM
if settings.OPENAI_API_KEY:
    llm = ChatOpenAI(
        api_key=settings.OPENAI_API_KEY,
        model=settings.OPENAI_MODEL,
        temperature=0,
    )
else:
    llm = ChatOllama(
        model="llama3.2:3b",
        temperature=0,
    )
    
class VectorStoreRetriever:
    def __init__(self, client: ClientAPI, embeddings: Embeddings, db: VectorStore, store: dict, retriever: ParentDocumentRetriever):
        self.client = client
        self.embeddings = embeddings
        self.db = db
        self.store = store
        self.retriever = retriever

    @staticmethod
    def load_documents_from_urls(urls: Dict[str, list]) -> List[Document]:
        documents = []
        # Set up Chrome options
        options = Options()
        options.add_argument('--no-sandbox')
        options.add_argument('--headless=new')
        driver = webdriver.Chrome(options=options)
        # Load each URL and extract text
        for cat, ls in urls.items():
            for url in ls:
                try:
                    driver.get(url)
                    strainer = bs4.SoupStrainer(name='div', attrs={'data-testid': 'RichText'})
                    soup = bs4.BeautifulSoup(driver.page_source, 'html.parser', parse_only=strainer)
                    documents.append(Document(
                        page_content=soup.get_text('\n', strip=True),
                        metadata={'category': cat, 'source': url},
                    ))
                except Exception as e:
                    print(f'Error loading {url}: {e}')
        driver.quit()
        return documents
    
    @classmethod
    def from_urls(cls, client: ClientAPI, embeddings: Embeddings, urls: Dict[str, list], child_db: str, parent_db: str):
        # Parent & child text splitters
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
        # Load documents from URLs
        documents = cls.load_documents_from_urls(urls)
        # Child chunk vectorstore
        client.get_or_create_collection(
            name=child_db,
            configuration={'hnsw': {'space': 'cosine'}},
        )
        db = Chroma(
            client=client,
            collection_name=child_db,
            embedding_function=embeddings,
        )
        # The storage layer for the parent documents
        try:
            store = joblib.load(f'./vectordb/{parent_db}.pkl')
        except FileNotFoundError:
            store = InMemoryStore()
        # Parent document retriever
        retriever = ParentDocumentRetriever(
            vectorstore=db,
            docstore=store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
        )
        # Add documents
        retriever.add_documents(documents)
        # Save parent store to disk
        joblib.dump(retriever.docstore, './vectordb/{parent_db}.pkl')
        return cls(client, embeddings, db, store, retriever)

    @classmethod
    def from_mem(cls, client: ClientAPI, embeddings: Embeddings, child_db: str, parent_db: str):
        # Parent & child text splitters
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
        # Child chunk vectorstore
        db = Chroma(
            client=client,
            collection_name=child_db,
            embedding_function=embeddings,
        )
        # Load parent store from disk
        store = joblib.load(f'./vectordb/{parent_db}.pkl')
        # Parent document retriever
        retriever = ParentDocumentRetriever(
            vectorstore=db,
            docstore=store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
        )
        return cls(client, embeddings, db, store, retriever)

    def query(self, query: str, cat: str, k: int) -> List[Document]:
        return self.retriever.invoke(query, search_kwargs={'k': k, 'where': {'category': cat}})

client = chromadb.PersistentClient(path='./vectordb')
embeddings = OllamaEmbeddings(model='nomic-embed-text')
retriever = VectorStoreRetriever.from_mem(client, embeddings, 'tncs', 'parent_store')

@tool
def lookup_tncs(query: str, category: Literal['mobile', 'broadband'], k: int = 5) -> str:
    """
    Look up terms and conditions for a given query and category.
    
    Args:
        query (str): The search query.
        category (str): The category to filter results by, either 'mobile' or 'broadband'.
        k (int): The number of results to return.
        
    Returns:
        str: A string containing the page content of the retrieved documents.
    """
    logging.info(f"Looking up T&Cs for query: {query}, category: {category}")
    docs = retriever.query(query, category, k)
    return "\n\n".join([doc.page_content for doc in docs])

# Examples for few-shot prompting
examples = [
    (
        "100 GB data; SIM only; at most $60 per month; at least 500 minutes talktime",
        MobilePlanRequest(local_data=100, plan_type="SIM only", price_monthly=60, talktime=500),
    ),
    (
        "caller ID; roaming data in Asia; 60GB data",
        MobilePlanRequest(caller_id=True, roam_data_region="Asia", local_data=60),
    ),
    (
        "with phone; 600 mins",
        MobilePlanRequest(plan_type="Phone", talktime=600),
    )
]
    
def extract_features(query: str, examples: list = examples) -> MobilePlanRequest:
    system_message = """
        Extract information from a semi-colon separated list of mobile plan features. Only include features that are explicitly mentioned. Leave values UNSET if not specified.
    """
    prompt = ChatPromptTemplate([
        ('system', system_message),
        MessagesPlaceholder('messages', optional=True),
        ('human', '{query}'),
    ])
    
    # Convert examples to messages
    messages = []
    for txt, tool_call in examples:
        messages.extend(
            tool_example_to_messages(txt, [tool_call])
        )
    llm_structured = prompt | llm.with_structured_output(MobilePlanRequest)
    return llm_structured.invoke({'query': query, 'messages': messages})

@tool
def get_recommendation(query: str) -> List[MobilePlan] | str:
    """
    Get a mobile plan recommendation based on the user's query.
    
    Args:
        query (str): User's requests are semi-colon separated. For example, '100 GB data; SIM only; at most $60 per month; at least 500 minutes talktime'.
        
    Returns:
        list: A list of mobile plans.
    """
    logging.info(f"Extracting features from query: {query}")
    request = extract_features(query)
    logging.info("Extracted results:", request.model_dump(exclude_unset=True))
    if result := recommender.recommend(request, k=2):
        return result
    else:
        return 'No suitable mobile plan found based on your request.'
