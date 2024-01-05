from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from ingest_data import embed_doc
from callback import MyCustomHandler
from route import dl


def get_chain(vectorstore):

    retriever = vectorstore.as_retriever()

    template = """Répond à la question en utilisant le contexte suivant:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    model = ChatOpenAI(callbacks=[MyCustomHandler()])

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    return chain

# vectorstore = embed_doc("data")

# chain = get_chain(vectorstore)

# question = "Quelle conférence Farid va présenter à Confoo?"

# #
# # ROUTAGE SÉMANTIQUE
# route = dl(question)

# if route.name is not None:
#     #
#     # ROUTE TROUVÉE, on parle d'un des sujets autorisés!, APPEL DU LLM AVEC LE CONTEXTE ET LA QUESTION
#     print(chain.invoke(question))
# else:
#     #
#     # AUCUNE ROUTE DE TROUVÉ ?
#     #    -> NE PAS APPELER LE LLM, faire une réponse toute faite!
#     print("Désolé, je ne répond qu'aux questions sur Confoo et Farid!")
