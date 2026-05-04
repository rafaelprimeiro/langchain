from langchain.tools import BaseTool
from dotenv import load_dotenv
from langchain_classic.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
import os

from pydantic import BaseModel, Field

load_dotenv()

class StudentExtractor(BaseModel):
    estudante: str = Field("Nome do estudante informado, sempre em letras minúsculas. Exemplo: joão, carlos, joana, carla")

class StudentDataTools(BaseTool):
    name: str = "DadosDeEstudante"
    description: str = """
    Esta ferramenta extrai o histórico e preferências de um estudante de acordo com seu histórico
"""
    def _run(self, input: str) -> str:
        llm = ChatOpenAI(model="gpt-5.4-nano",
                         api_key=os.getenv("OPENAI_API_KEY"))
        parser = JsonOutputParser(pydantic_object=StudentExtractor)
        template = PromptTemplate.from_template(
            template="""Você deve analisar a {input} para extrair o nome de usuário informado.
            Formato de saída: {formato_saida}""",
            partial_variables={"formato_saida": parser.get_format_instructions()})
        
        chain = template | llm | parser
        response = chain.invoke({"input": input})
        print(response)
        return response['estudante']
        
question = "Quais são os dados da Ana"

response = StudentDataTools()._run(question)
print(response)
