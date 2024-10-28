import streamlit as st
from streamlit_chat import message
import os
from ingest_data import embed_doc
import openai
from route import dl
from kownledge import get_chain
from langchain_core.tools import Tool
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain

openai.api_key = os.getenv("OPENAI_API_KEY")

title = "Bot KB Demo (semantic router)"
st.set_page_config(page_title=title, page_icon=":shark:")
st.header(title)

# Initialize the vector store on first load
if "vectorstore" not in st.session_state:
    with st.spinner("Vector Database: création de la base de connaissance..."):
        vectorstore = embed_doc("data")
        st.session_state["vectorstore"] = vectorstore
        
        search = GoogleSearchAPIWrapper()
        
        def top_result(query):
            result = search.results(query=query, num_results=1)
            
            if result:  # Check if there's at least one result
                top_result = result[0]  # Get the first result
                # Transform the first result to a mapping
                return {
                    "title": top_result.get("title"),
                    "snippet": top_result.get("snippet"),
                    "link": top_result.get("link")
                }
            else:
                return None  # Return None if no results are found
    
        search_tool = Tool(
            name="google_search",
            description="Search Google for recent results.",
            func=top_result,
        )
        st.session_state["search_tool"] = search_tool

        st.write("Base de connaissances créée avec succès!")

# Initialize the state for the last question and answer
if "last_question" not in st.session_state:
    st.session_state["last_question"] = ""
if "last_answer" not in st.session_state:
    st.session_state["last_answer"] = ""
if "last_retrieved_docs" not in st.session_state:
    st.session_state["last_retrieved_docs"] = []

# Create an empty placeholder for the input
placeholder = st.empty()

# Function to get user input
def get_text():
    input_text = placeholder.text_input("You: ", "", key="input", on_change=submit_input)
    return input_text

# Function to submit input
def submit_input():
    user_input = st.session_state["input"]

    if user_input and st.session_state["vectorstore"]:
        vectorstore = st.session_state["vectorstore"]
        chain = get_chain(vectorstore)

        # Clear previous responses when a new question is asked
        st.session_state["last_question"] = user_input
        st.session_state["last_answer"] = ""
        st.session_state["last_retrieved_docs"] = []

        # Route the question
        route = dl(user_input)
        print(route)

        if route.name == 'joke':
            # Use OpenAI gpt4-mini model to generate a joke
            
            llm = ChatOpenAI( 
                model="gpt-4o",
                temperature=1,
                max_tokens=None,
                timeout=None,
                max_retries=2,
            )
            
            messages = [
                (
                    "system",
                    "Tu es un humouriste reconnu pour tes blagues tres drole en francais. Tu adores les jeux de mots. Trouves une blague correspondant au theme.",
                ),
                ("human", user_input),
            ]
            output = llm.invoke(messages)

            st.session_state["last_answer"] = output.content
        elif route.name is not None:
            print(route.name)
            # Get answer from the LLM
            output = chain.invoke(user_input)
            st.session_state["last_answer"] = output["answer"]
            st.session_state["last_retrieved_docs"] = output["retrieved_docs"]
        else:
            # use search tool (google) to reply
            search_tool = st.session_state["search_tool"]
            
            output = search_tool.run(user_input)
            
            if output:
                # Use structured output for display
                formatted_output = f"**{output['title']}**\n\n{output['snippet']}\n\n[Learn more]({output['link']})"
                st.session_state["last_answer"] = formatted_output
            else:
                st.session_state["last_answer"] = "No relevant search results found."


        # Clear input after submission
        st.session_state["input"] = ""

# Get user input (this will trigger on_change with the submit_input function)
get_text()

# Display the last question and answer
if st.session_state["last_question"]:
    # Display the user's last message
    message(st.session_state["last_question"], is_user=True, key="last_user", avatar_style='big-smile')

    # Display the assistant's last answer
    message(st.session_state["last_answer"], key="last_assistant", avatar_style='adventurer')

    # Display the last references, if any
    last_retrieved_docs = st.session_state["last_retrieved_docs"]
    if last_retrieved_docs:
        st.markdown("**Références :**")
        for j, doc in enumerate(last_retrieved_docs):
            source = doc.metadata.get('source', 'Source inconnue')
            with st.expander(f"[{j + 1}] {source}"):
                st.write(doc.page_content)

