"""
Bot KB Démo
"""

import streamlit as st
from streamlit_chat import message
import os
from ingest_data import embed_doc
import openai
from query_data import get_chain, QA_PROMPT, CONDENSE_QUESTION_PROMPT, _template
import pickle

openai.api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Bot KB Demo", page_icon=":shark:")
st.header("Bot KB Demo")

if "vectorstore" not in st.session_state:
    with st.spinner("Vector Database: création de la base de connaissance..."):
        vectorstore = embed_doc("data")
        st.session_state["vectorstore"] = vectorstore
        st.write("Base de connaissances créée avec succès!")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

placeholder = st.empty()

def get_text():
    input_text = placeholder.text_input("You: ", "", key="input")
    return input_text

user_input = get_text()

if "vectorstore" in st.session_state:
    vectorstore = st.session_state["vectorstore"]

if st.button("Soumettre question") and vectorstore is not None:

    chain = get_chain(vectorstore)

    #
    # COSINE SIMILARITY -> RECHERCHE DANS LE VECTOR STORE
    #
    docs = vectorstore.similarity_search(user_input)

    if user_input:
        
        #
        # APPEL A OPENAI AVEC LE PROMTP CONTENANT les 2 DOCUMENTS TROUVÉS ET LA QUESTION
        #
        output = chain.run(input=user_input, vectorstore=vectorstore, context=docs[:2], chat_history=[], question=user_input, 
                           QA_PROMPT=QA_PROMPT, CONDENSE_QUESTION_PROMPT=CONDENSE_QUESTION_PROMPT, template=_template)

        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)

        print(st.session_state.generated)

    placeholder = ""

if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"]) -1, -1, -1):
        message(st.session_state["generated"][i], key=str(i), avatar_style='adventurer')
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user", avatar_style='big-smile')
