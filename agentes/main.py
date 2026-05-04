import json

from langchain.tools import BaseTool
from dotenv import load_dotenv
from langchain_classic.agents import Tool, create_openai_tools_agent, AgentExecutor
from langchain_classic.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
import os
import pandas as pd

from pydantic import BaseModel, Field

load_dotenv()

def find_student_data(student_name: str) -> dict:
    datas = pd.read_csv("documentos/estudantes.csv")
    student_data = datas[datas['USUARIO'] == student_name]
    if student_data.empty:
        return {}
    return student_data.iloc[:1].to_dict()

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
        studant_data = find_student_data(student)
        return json.dumps(studant_data)
        

llm = ChatOpenAI(model="gpt-5.4-nano",
                 api_key=os.getenv("OPENAI_API_KEY"))
student_data_tool = StudentDataTools()

question = "Quais são os dados da Bianca"

tools = [
    Tool(
        name=student_data_tool.name, 
        func=student_data_tool._run, 
        description=student_data_tool.description)
]

prompt = ChatPromptTemplate.from_messages([
    ("system", "Você é um assistente legal."),
    ("user", "{input}\n\n{agent_scratchpad}")
])

agent = create_openai_tools_agent(llm, tools, prompt)

executer = AgentExecutor(agent=agent, tools=tools, verbose=True)

response = executer.invoke({"input": question})

print(response)

# response = StudentDataTools()._run(question)
# print(response)
