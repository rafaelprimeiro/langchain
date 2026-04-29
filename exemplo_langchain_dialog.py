from langchain_core.prompts import ChatPromptTemplate

# Criando um template de chat para um cenário de RPG
chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Você é o narrador de uma aventura de RPG."),
        ("human", "Me conte sobre a cidade que estamos explorando."),
        ("ai", "Você está em Eldoria, uma cidade antiga conhecida por suas ruínas místicas e mercados movimentados."),
        ("human", "Quero saber mais sobre o templo principal."),
        ("ai", "O Templo de Solara é o coração espiritual de Eldoria, famoso por seu vasto acervo de relíquias sagradas e histórias antigas.")
    ]
)

# Formatando as mensagens para um diálogo específico
dialogo = chat_template.format_messages()
print(dialogo)