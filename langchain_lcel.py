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

llm = ChatOpenAI(model="gpt-5.4-nano", 
                 temperature=0.7, 
                 openai_api_key=api_key)

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
    "Sugira um restaurante para jantar entre locais {cidade}."
)

cuture_model = ChatPromptTemplate.from_template(
    "Sugira locais e atividade cultural para fazer durante {cidade}."
)

city_chain = city_model | llm | parser
dinner_chain = dinner_model | llm | StrOutputParser()
culture_chain = cuture_model | llm | StrOutputParser()

chain = city_chain | {
    "restaurantes": dinner_chain,
    "locais_culturais": culture_chain
}

print(chain.invoke(interesse))