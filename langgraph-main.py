from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig
from typing import TypedDict, Literal
import asyncio

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

class State(TypedDict):
    query: str
    destiny: Rota
    response: str

async def no_roteador(state: State, config: RunnableConfig):
    return {"destiny": await roteador.ainvoke({"query": state["query"]}, config=config)}

async def no_praia(state: State, config: RunnableConfig):
    return {"response": await cadeia_praia.ainvoke({"query": state["query"]}, config=config)}

async def no_montanha(state: State, config: RunnableConfig):
    return {"response": await cadeia_montanha.ainvoke({"query": state["query"]}, config=config)}

def escolher_no(state: State):
    return "praia" if state["destiny"]["destino"] == "praia" else "montanha"

graph = StateGraph(State)
graph.add_node("roteador", no_roteador)
graph.add_node("praia", no_praia)
graph.add_node("montanha", no_montanha)

graph.add_edge(START, "roteador")
graph.add_conditional_edges("roteador", escolher_no)

graph.add_edge("praia", END)
graph.add_edge("montanha", END)

app = graph.compile()

async def main():
    pergunta = "Quero visitar um lugar famoso no Brasil por praias e culturas."
    result = await app.ainvoke({"query": pergunta})
    print(f"Pergunta: {pergunta}")
    print(f"Resposta: {result['response']}")

if __name__ == "__main__":
    asyncio.run(main())
