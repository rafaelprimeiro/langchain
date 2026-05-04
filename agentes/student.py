import json

from langchain_classic.prompts import PromptTemplate
from langchain_classic.tools import BaseTool
from langchain_openai import ChatOpenAI
import pandas as pd
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
import os

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
        student = response['estudante']
        student = student.lower()
        studant_data = self.find_student_data(student)
        return json.dumps(studant_data)
    
    def find_student_data(self, student_name: str) -> dict:
        datas = pd.read_csv("documentos/estudantes.csv")
        student_data = datas[datas['USUARIO'] == student_name]
        if student_data.empty:
            return {}
        return student_data.iloc[:1].to_dict()