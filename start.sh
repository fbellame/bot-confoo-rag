#!/bin/bash

export OPENAI_API_KEY=
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
export LANGCHAIN_API_KEY=
export LANGCHAIN_PROJECT="semantic-router"
export GOOGLE_API_KEY=
export GOOGLE_CSE_ID=

streamlit run server.py