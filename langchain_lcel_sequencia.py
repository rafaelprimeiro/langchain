from langchain_core.runnables import RunnablePassthrough
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

part_one = PromptTemplate.from_template("Analisar a queixa: {queixa}") | llm | StrOutputParser()
part_two = PromptTemplate.from_template("Avaliar sentimento da queixa: {resultado_analise}") | llm | StrOutputParser()
part_tree = PromptTemplate.from_template("Formular resposta: {sentimento}") | llm | StrOutputParser()


chian = (
    {"queixa": RunnablePassthrough()}
    | RunnablePassthrough.assign(resultado_analise=part_one)
    | RunnablePassthrough.assign(sentimento=part_two)
    | part_tree
)

# Neste exemplo:
# - **`PromptTemplate`**: define os prompts para cada etapa da cadeia.
# - **`|` (operador de pipe)**: encadeia componentes, passando a saída de um como entrada para o próximo.
# - **`RunnablePassthrough`**: utilizado para mapear e passar dados através da cadeia.
# - **`StrOutputParser`**: converte a saída do modelo em strings.
# Este exemplo destaca como LCEL permite compor cadeias complexas de maneira intuitiva e flexível, integrando diversos componentes do LangChain.

queixa = "O produto chegou com defeito e o atendimento ao cliente foi ruim."
response = chian.invoke(queixa)

print(response)