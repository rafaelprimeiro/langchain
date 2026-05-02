from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.globals import set_debug
from pydantic import Field, BaseModel

from dotenv import load_dotenv
import os
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
set_debug(True)

llm = ChatOpenAI(model="gpt-5.4-nano", 
                 temperature=0.7, 
                 openai_api_key=api_key)


messages = [
    ("human", "Qual é a capital da França?"),
    # ("ai", "A capital da França é Paris."),
    ("human", "Qual é a capital da Alemanha?"),
    # ("ai", "A capital da Alemanha é Berlim."),
    ("human", "Quais são os paises da américa do sul?"),
    # ("ai", "Os países da América do Sul são: Argentina, Bolívia, Brasil, Chile, Colômbia, Equador, Guiana, Paraguai, Peru, Suriname, Uruguai e Venezuela."),
    ("human", "Quais paises não fazem fronteira com o Brasil?"),
    # ("ai", "Os países que não fazem fronteira com o Brasil são: Chile, Equador, Guiana, Suriname e Uruguai.")
]

store = {}  # mantém histórico por session_id

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

conversation = RunnableWithMessageHistory(llm, get_session_history)

for role, message in messages:
    resp = conversation.invoke((role, message),
                                config={"configurable": {"session_id": "user-123"}})
    print(resp)