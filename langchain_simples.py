from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

numero_de_dias = 7
numero_de_criancas = 2
atividade = "praia"

promptTemplate = PromptTemplate.from_template(
    "Crie um roteiro de viagem de {dias} dias, para uma família com {criancas} crianças, que gostam de {atividade}."    
)

prompt = promptTemplate.format(dias=numero_de_dias,
                               criancas=numero_de_criancas,
                               atividade=atividade
)
print(prompt)

llm = ChatOpenAI(model="gpt-5.4-nano", temperature=0.7, openai_api_key=api_key)

response = llm.invoke(prompt)

print(response.content)
