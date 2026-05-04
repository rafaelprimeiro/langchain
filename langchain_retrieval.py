from langchain_classic.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.globals import set_debug
from langchain_text_splitters import CharacterTextSplitter
from langchain_classic.document_loaders import TextLoader

from dotenv import load_dotenv
import os
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
set_debug(True)

llm = ChatOpenAI(model="gpt-5.4-nano", 
                 temperature=0.7, 
                 openai_api_key=api_key)

textLoader = TextLoader("Contrato_Itau.md", encoding="utf-8")
documents = textLoader.load()

splitter = CaracterTextSplitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
split_document = splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)

# Isso tem custo. O ideal é armazenar os embeddings 
# em um banco de dados para não precisar recalcular toda vez.
db = FAISS.from_documents(split_document, embeddings)

qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())

question = "Como devo procede quando tenho um problema com o cartão de crédito?"

answer = qa_chain.invoke({"query": question})
print(answer)