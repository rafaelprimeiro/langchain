import json
from typing import List

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
    estudante: str = Field("Nome do estudante informado, sempre em letras minúsculas. Exemplo: joão, carlos, joana, carla. Esta ferramenta é apenas capaz de extrair um nome de estudante por vez. Se houver mais de um nome, o resultado será indefinido.")

class StudentDataTools(BaseTool):
    name: str = "DadosDeEstudante"
    description: str = """
    Esta ferramenta extrai o histórico e preferências de um estudante de acordo com seu histórico. 
    Esta extração é feita a partir de um nome de estudante informado. Esta ferramenta não é capaz de extrair o nome do estudante, 
    apenas de buscar os dados do estudante a partir do nome.
"""
    def _run(self, input: str) -> str:
        student = input.lower().strip()
        studant_data = self.find_student_data(student)
        return json.dumps(studant_data)
    
    def find_student_data(self, student_name: str) -> dict:
        datas = pd.read_csv("documentos/estudantes.csv")
        student_data = datas[datas['USUARIO'] == student_name]
        if student_data.empty:
            return {}
        return student_data.iloc[:1].to_dict()
    
class Grade(BaseModel):
    subject: str = Field("Nome da disciplina cursada pelo estudante")
    grade: float = Field("Nota obtida na disciplina cursada pelo estudante")
    
class AcademicProfileExtractor(BaseModel):
    name: str = Field("Nome do estudante informado")
    conclusion_year: int = Field("Ano de conclusão do curso do estudante informado")
    grades: List[Grade] = Field("Lista de disciplinas cursadas pelo estudante, com nome da disciplina e nota (score/grade)")
    summary: str = Field("Resumo do perfil acadêmico do estudante, destacando pontos fortes e fracos, interesses e preferências")

class AcademicProfileTool(BaseTool):
    name: str = "PerfilAcademico"
    description: str = """Esta ferramenta cria um perfil acadêmico de um estudante a partir de seus dados. 
    Esta ferramenta não é capaz de buscar os dados do estudante."""
    def _run(self, input: str) -> str:
        llm = ChatOpenAI(model="gpt-5.4-nano",
                         api_key=os.getenv("OPENAI_API_KEY"))
        parser = JsonOutputParser(pydantic_object=AcademicProfileExtractor)
        template = PromptTemplate(
            template=""" - Formate o estudante para seu perfil acadêmico.
            - Com os dados, identifique as opções de universidades sugeridas e cursos compatíveis com o interesse do aluno
            - Destaque o perfil do aluno dando enfase principalmente naquilo que faz sentido para as instituições de interesse do aluno

            Persona: você é uma consultora de carreira e precisa indicar com detalhes, riqueza, mas direta ao ponto para o estudante as opções e consequências possíveis.
            Informações atuais:

            {dados_do_estudante}
            {formato_de_saida}
            """,
            input_variables=["dados_do_estudante"],
            partial_variables={"formato_de_saida": parser.get_format_instructions()}
        )
        chain = template | llm | parser
        response = chain.invoke({"dados_do_estudante": input})
        return response