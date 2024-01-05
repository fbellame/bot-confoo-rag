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

