from langchain_openai import ChatOpenAI
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_core.globals import set_debug

from dotenv import load_dotenv
import os
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
set_debug(True)

llm = ChatOpenAI(model="gpt-5.4-nano", 
                 temperature=0.7, 
                 openai_api_key=api_key)

examples = [
    {
        "question": "Quem viveu mais, Muhammad Ali ou Alan Turing?",
        "answer": """
        São necessárias perguntas de acompanhamento: Sim.
        Pergunta: Quantos anos Muhammad Ali tinha quando morreu?
        Resposta intermediária: Muhammad Ali tinha 74 anos quando morreu.
        Pergunta: Quantos anos Alan Turing tinha quando morreu?
        Resposta intermediária: Alan Turing tinha 41 anos quando morreu.
        Então a resposta final é: Muhammad Ali
        """,
    },
    # Outros exemplos aqui...
]

example_prompt = PromptTemplate(
    input_variables=["question", "answer"], template="Pergunta: {question}\n{answer}"
)

prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Pergunta: {input}",
    input_variables=["input"],
)

prompt = prompt_template.format(input="Quem foi o pai de Mary Ball Washington?")
resposta = llm.invoke(prompt)
print(resposta.content)