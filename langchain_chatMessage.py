from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.globals import set_debug

from dotenv import load_dotenv
import os
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
# set_debug(True)

llm = ChatOpenAI(model="gpt-5.4-nano", 
                 temperature=0.7, 
                 openai_api_key=api_key)


prompt_sugestion = [
    ("system", "Você é um assistente de viagens especializado em destinos no Brasil, oferecendo recomendações personalizadas com base nas preferências do usuário."),
    ("placeholder", "{history}"),
    ("human", "{query}"),
]

questions_list = [
        "Quero visitar um lugar no Brasil famoso por suas praias e cultura. Pode me recomendar?",
        "Qual é o melhor período do ano para visitar em termos de clima?",
        "Quais tipos de atividades ao ar livre estão disponíveis?",
        "Alguma sugestão de acomodação eco-friendly por lá?",
        "Cite outras 20 cidades com características semelhantes às que descrevemos até agora. Rankeie por mais interessante, incluindo no meio a que você já sugeriu.",
        "Na primeira cidade que você sugeriu lá atrás, quero saber 5 restaurantes para visitar. Responda somente o nome da cidade e o nome dos restaurantes.",
]

store = {}  # mantém histórico por session_id

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

conversation = RunnableWithMessageHistory(
    runnable=llm,
    get_session_history=get_session_history,
    input_message_key="query",
    history_message_key="history",
)

for message in questions_list:
    resp = conversation.invoke(
        {
            "query": message
        },
        config={
            "configurable": {
                "session_id": "user-123"
            }
        }
    )
    print("💬 Usuário:", message)
    print("🤖 Assistente:", resp)

print("🛠️ - ", store["user-123"].messages)