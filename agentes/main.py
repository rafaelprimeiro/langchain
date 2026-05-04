from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor
from langchain_openai import ChatOpenAI
import os

from agent import AgentOpenAIFunctions

load_dotenv()

llm = ChatOpenAI(model="gpt-5.4-nano",
                 api_key=os.getenv("OPENAI_API_KEY"))

question = "Quais são os dados da Ana e da Bianca?"

agent = AgentOpenAIFunctions(llm)

executer = AgentExecutor(agent=agent.agent, 
                         tools=agent.tools, 
                         verbose=True)

response = executer.invoke({"input": question})

print(response)

# response = StudentDataTools()._run(question)
# print(response)
