"""
Bot KB Démo
"""

import streamlit as st
from streamlit_chat import message
import os
from ingest_data import embed_doc
import openai
from langchain.chains import LLMChain
from route import dl
from kownledge import get_chain

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
    chain: LLMChain

    chain = get_chain(vectorstore)

    if user_input:
        question = user_input

        #
        # ROUTAGE SÉMANTIQUE
        route = dl(question)

        if route.name is not None:
            print(route.name)
            #
            # ROUTE TROUVÉE, on parle d'un des sujets autorisés!, APPEL DU LLM AVEC LE CONTEXTE ET LA QUESTION
            output = chain.invoke(question)
        else:
            #
            # AUCUNE ROUTE DE TROUVÉ ?
            #    -> NE PAS APPELER LE LLM, faire une réponse toute faite!
            output = "Désolé, je ne répond qu'aux questions sur Confoo et Farid!"

        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)

        print(st.session_state.generated)

    placeholder = ""

if "generated" in st.session_state and st.session_state["generated"]:
    for i in range(len(st.session_state["generated"]) -1, -1, -1):
        message(st.session_state["generated"][i], key=str(i), avatar_style='adventurer')
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user", avatar_style='big-smile')
