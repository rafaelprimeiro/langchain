# Regras para Chat (regras_chat.md)

## Resumo
Breve guia sobre o que são "rules" (regras) em prompts de chat, por que usar, e exemplos práticos com `ChatPromptTemplate.from_messages`.

## O que são (definição)
- Regras: instruções de comportamento incluídas como mensagens de papel `system` (mensagens de sistema) que orientam o LLM sobre tom, formato, limites e restrições.
- Implementação: normalmente uma string longa dentro de uma `SystemMessage` no `ChatPromptTemplate` ou injetada via `partial_variables`.

## Vantagens principais
- **Consistência**: mantém estilo e formato entre chamadas.
- **Segurança**: centraliza restrições (ex.: "não invente fatos").
- **Reutilização**: regras separadas da entrada do usuário tornam templates reaproveitáveis.
- **Composição**: combinam bem com `MessagesPlaceholder`, `partial_variables` e `OutputParsers`.
- **Testabilidade**: fácil inspeção antes do envio (`format_prompt(...).to_messages()`).

## Como usar com `ChatPromptTemplate.from_messages`
- Forneça uma sequência de representações de mensagem: tuplas `(role, template)`, instâncias de `BaseMessagePromptTemplate`, ou placeholders.
- Coloque as regras na mensagem com papel `system` (ou `system`-like) no início para maior prioridade.
- Use variáveis (`{var}`) nas mensagens e preencha com `format_prompt(...)`.
- Combine com `JsonOutputParser` / `pydantic` para forçar e validar formato de saída.
- Use `MessagesPlaceholder` para injetar histórico de conversas dinamicamente.

## Exemplo (prático, adaptável ao seu projeto)
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
import os

# modelo de saída validado
class Destino(BaseModel):
    cidade: str = Field(description="Cidade a visitar")
    motivo: str = Field(description="Por que visitar")

parser = JsonOutputParser(pydantic_object=Destino)

chat_template = ChatPromptTemplate.from_messages(
    [
        ("system",
         "Regras:\n"
         "1) Responda em Português.\n"
         "2) Seja conciso (máx 2 parágrafos).\n"
         "3) Quando solicitado, retorne JSON válido conforme o schema."),
        ("user", "Sugira uma cidade para quem gosta de {interesse}."),
    ]
)

# preenche variáveis e injeta instrução de formato (do parser)
prompt_value = chat_template.format_prompt(interesse="gastronomia",
                                          output_format=parser.get_format_instructions())

messages = prompt_value.to_messages()

llm = ChatOpenAI(temperature=0.0, openai_api_key=os.getenv("OPENAI_API_KEY"))
# envio ao LLM (API pode variar por integração)
response = llm(messages=messages)
# se usar JsonOutputParser: parse do texto de saída para pydantic
# parsed = parser.parse(response)  # método ilustrativo dependendo da integração
```

## Boas práticas
- Coloque regras críticas no topo do template (`system`) para maior prioridade.
- Especifique formato esperável (ex.: JSON + schema) e combine com `OutputParser` para validação automática.
- Use `partial_variables` para instruções repetitivas (ex.: `partial_variables={"output_format": parser.get_format_instructions()}`).
- Teste sempre com `format_prompt(...).to_messages()` antes de enviar para garantir ordem e conteúdo.