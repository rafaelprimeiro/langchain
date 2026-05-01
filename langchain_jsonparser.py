from langchain_classic.output_parsers import DatetimeOutputParser
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

class Destino(BaseModel):
    cidade: str = Field(description="Cidade a visitar")
    motivo: str = Field(description="Motivo pelo qual é interessante visitar a cidade")

parser = JsonOutputParser(pydantic_object=Destino)

interesse = "praia"

city_model = PromptTemplate(
    template=""""Sugira uma cidade dado meu interesse por {interesse}.
    {output_format}
    """,
    input_variables=["interesse"],
    partial_variables={"output_format": parser.get_format_instructions()}
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

# print(chain.invoke(interesse))

# Exemplo de uso do parser de saída para extrair informações de data e hora

parseador_saida = DatetimeOutputParser()
modelo_data = """Responda a pergunta do usuário: 
    {pergunta}

    {formato_saida}
"""

prompt = PromptTemplate.from_template(
    modelo_data,
    partial_variables={"formato_saida": parseador_saida.get_format_instructions()},
)

cobaia = PromptTemplate(
    input_variables=['pergunta'], 
    partial_variables={'formato_saida': "Write a datetime string that matches the following pattern: '%Y-%m-%dT%H:%M:%S.%fZ'.\n\nExamples: 0668-08-09T12:56:32.732651Z, 1213-06-23T21:01:36.868629Z, 0713-07-06T18:19:02.257488Z\n\nReturn ONLY this string, no other words!"},
    template='Answer the users question:\n\n{pergunta}\n\n{formato_saida}')

chain = prompt | llm | parseador_saida

resposta = chain.invoke({"pergunta": "Quando a bitcoin foi fundada?"})

print(resposta)