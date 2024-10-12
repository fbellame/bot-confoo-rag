from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from callback import MyCustomHandler

def get_chain(vectorstore):

    retriever = vectorstore.as_retriever()

    template = """Répond à la question en utilisant le contexte suivant:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    model = ChatOpenAI(model_name="gpt-4o-mini", callbacks=[MyCustomHandler()])

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    return chain

