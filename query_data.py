from langchain.prompts.prompt import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import ChatVectorDBChain

from callback import MyCallbackHandler
from langchain.callbacks.base import CallbackManager

#########################################################
#
# PROMTP ENGEREERING
#
_template = """Compte tenu de la conversation suivante et d'une question de suivi, reformulez la question de suivi pour en faire une question autonome.
Historique des discussions:
{chat_history} 
Entrée de suivi: {question}
Question autonome:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

template = """Vous êtes un assistant aux questions sur la présentation de Farid Bellameche à Confoo. Vous ne répondez qu'en FRANCAIS SVP!!!
On vous donne les parties extraites suivantes d'un long document et une question. Fournissez une réponse conversationnelle.
Si vous ne connaissez pas la réponse, dites simplement "Hmm, je ne répond qu'aux question concernant Confoo et Farid". N'essayez pas d'inventer une réponse.
Si la question ne porte pas sur la présentation de Farid Bellameche à Confoo, informez-les poliment que vous ne répondez qu'aux questions concernant Confoo et Farid.

Question: {question}
=========
{context}
=========
Réponse en Markdown:"""
QA_PROMPT = PromptTemplate(template=template, input_variables=["question", "context"])

#########################################################


def get_chain(vectorstore):

    manager = CallbackManager([MyCallbackHandler()])

    llm = OpenAI(temperature=0, callback_manager=manager, model_name="gpt-3.5-turbo")
    qa_chain = ChatVectorDBChain.from_llm(
        llm,
        vectorstore,
        qa_prompt=QA_PROMPT,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT
    )
    return qa_chain