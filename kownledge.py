from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def get_chain(vectorstore):

    retriever = vectorstore.as_retriever()

    template = """Répond à la question en utilisant le contexte suivant:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    model = HuggingFacePipeline.from_model_id(
                model_id="mistralai/Mistral-7B-Instruct-v0.2",
                task="text-generation",
                device=0,  # replace with device_map="auto" to use the accelerate library.
                pipeline_kwargs={"max_new_tokens": 200},
)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    return chain

