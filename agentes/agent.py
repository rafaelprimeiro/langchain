from langchain_classic.agents import Tool, create_openai_tools_agent
from langchain_classic.prompts import ChatPromptTemplate
from dotenv import load_dotenv

from student import AcademicProfileTool, StudentDataTools

load_dotenv()

class AgentOpenAIFunctions:
    def __init__(self, llm):
        self.llm = llm
        student_data_tool = StudentDataTools()
        perfil_academico_tool = AcademicProfileTool()
        self.tools = [
            Tool(
                name=student_data_tool.name, 
                func=student_data_tool._run, 
                description=student_data_tool.description),
            Tool(
                name=perfil_academico_tool.name, 
                func=perfil_academico_tool._run, 
                description=perfil_academico_tool.description)
        ]

        prompt = ChatPromptTemplate.from_messages([
            ("system", "Você é um assistente legal."),
            ("user", "{input}\n\n{agent_scratchpad}")
        ])
        self.agent = create_openai_tools_agent(llm, self.tools, prompt)