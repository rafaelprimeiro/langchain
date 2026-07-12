from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import TypedDict, Literal

from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

modelo = ChatOpenAI(
    model="gpt-5.4-nano", 
    temperature=0.7, 
    openai_api_key=api_key)

prompt_consultor_praia = ChatPromptTemplate.from_messages([
    ("system", "Apresente-se como Sra Praia. Você é um assistente de viagens especializado em destinos para praias."),
    ("human", "{query}"),
])

prompt_consultor_montanha = ChatPromptTemplate.from_messages([
    ("system", "Apresente-se como Sr Montanha. Você é um assistente de viagens especializado em destinos para montanhas e atividades radicais."),
    ("human", "{query}"),
])

cadeia_praia = prompt_consultor_praia | modelo | StrOutputParser()
cadeia_montanha = prompt_consultor_montanha | modelo | StrOutputParser()

class Rota(TypedDict):
    destino: Literal["praia", "montanha"]

prompt_roteador = ChatPromptTemplate.from_messages([
    ("system", "Responda apenas com `praia` ou `montanha` com base na consulta do usuário."),
    ("human", "{query}"),
])

roteador = prompt_roteador | modelo.with_structured_output(Rota)

def responda(pergunta: str):
    rota = roteador.invoke({"query": pergunta})
    print(f"Rota sugerida: {rota['destino']}")
    if rota["destino"] == "praia":
        return cadeia_praia.invoke({"query": pergunta})
    return cadeia_montanha.invoke({"query": pergunta})
    
print(responda("Quero surfar em um lugar quente."))