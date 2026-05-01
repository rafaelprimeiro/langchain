from operator import itemgetter

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

class Destino(BaseModel):
    cidade: str = Field(description="Cidade a visitar")
    motivo: str = Field(description="Motivo pelo qual é interessante visitar a cidade")

parser = JsonOutputParser(pydantic_object=Destino)

city_model = PromptTemplate(
    template="""Sugira uma cidade dado meu interesse por {interesse}.
    {output_format}
    """,
    input_variables=["interesse"],
    partial_variables={"output_format": parser.get_format_instructions()},
)

dinner_model = ChatPromptTemplate.from_template(
    "Sugira restaurantes populares entre locais em {cidade}"
)

culture_model = ChatPromptTemplate.from_template(
    "Sugira atividades e locais culturais em {cidade}"
)

final_model = ChatPromptTemplate.from_messages(
    [
        ("ai", "Sugestão de viagem para a cidade: {cidade}"),
        ("ai", "Restaurantes que você não pode perder: {restaurantes}"),
        ("ai", "Atividades e locais culturais recomendados: {locais_culturais}"),
        ("system", "Combine as informações anteriores em 2 parágrafos coerentes")
    ]
)

part_one = city_model | llm | parser
part_two = dinner_model | llm | StrOutputParser()
part_three = culture_model | llm | StrOutputParser()
part_four = final_model | llm | StrOutputParser()

chain = part_one | {
    "restaurantes": part_two,
    "locais_culturais": part_three,
    "cidade" : itemgetter("cidade")
} | part_four

result = chain.invoke({"interesse": "gastronomia"})

print(result)