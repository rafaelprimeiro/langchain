from dotenv import load_dotenv
from langchain_core.globals import set_debug
from langchain_classic.agents import AgentExecutor
from langchain_openai import ChatOpenAI
import os

from agent import AgentOpenAIFunctions

set_debug(False)
load_dotenv()

llm = ChatOpenAI(model="gpt-5.4-nano",
                 api_key=os.getenv("OPENAI_API_KEY"))

question = "Quais os dados de Ana?"
question = "Quais os dados de Bianca?"
question = "Quais os dados de Ana e da Bianca?"
question = "Crie um perfil acadêmico para a Ana!"
question = "Compare o perfil acadêmico da Ana com o da Bianca!"
question = "Tenho sentido Ana desanimada com cursos de matemática. Seria uma boa parear ela com a Marcos?"

agent = AgentOpenAIFunctions(llm)

# executer = AgentExecutor(agent=agent.agent, 
#                          tools=agent.tools, 
#                          verbose=True)

# response = executer.invoke({"input": question})
# print(response)



# response = StudentDataTools()._run(question)
# print(response)

inputs = {"messages": [{"role": "user", "content": question}]}
result = agent.agent.invoke(inputs)
print(result)

# for chunk in agent.agent.stream(inputs, stream_mode="updates"):
#     print("🛠️", chunk)
