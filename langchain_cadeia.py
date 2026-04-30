from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.globals import set_debug

from dotenv import load_dotenv
import os
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
set_debug(True)

interesse = "praia"

city_model = ChatPromptTemplate.from_template(
    "Sugira uma cidade dado meu interesse por {interesse}. A sua resposta dever ser **SOMENTE** o nome da cidade, sem explicações ou detalhes adicionais."
)

dinner_model = ChatPromptTemplate.from_template(
    "Sugira um restaurante para jantar entre locais na {cidade}."
)

cuture_model = ChatPromptTemplate.from_template(
    "Sugira locais e atividade cultural para fazer durante {cidade}."
)

llm = ChatOpenAI(model="gpt-5.4-nano", 
                 temperature=0.7, 
                 openai_api_key=api_key)

city_chain = city_model | llm | StrOutputParser()
dinner_chain = dinner_model | llm | StrOutputParser()
culture_chain = cuture_model | llm | StrOutputParser()

chain = city_chain | dinner_chain | culture_chain

print(chain.invoke(interesse))