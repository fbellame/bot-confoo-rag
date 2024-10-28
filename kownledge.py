from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.vectorstores.base import VectorStore
from langchain.chains.base import Chain
from langchain.schema import Document
from typing import List, Dict, Any
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank

# FUNCTION TO INITIALIZE AND RETURN AN INSTANCE OF THE CHAIN
def get_chain(vectorstore: VectorStore):
    retriever = vectorstore.as_retriever()
    template = """Réponds à la question de manière succinte en utilisant le contexte suivant:
{context}

Question: {question}
Réponse (avec références):"""
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(model_name="gpt-4o-mini")
    chain = RetrievalWithCitationsChain(retriever=retriever, model=model, prompt=prompt)
    return chain

class RetrievalWithCitationsChain(Chain):
    retriever: Any
    model: Any
    prompt: ChatPromptTemplate
    
    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        question = inputs["question"]
        
        # RETRIEVE RELEVANT DOCUMENTS USING THE RETRIEVER
        compressor = FlashrankRerank(top_n=3)
        retrieved_docs: List[Document] = self.retriever.invoke(question)
        compressed_docs = compressor.compress_documents(retrieved_docs, question)
        # Build context
        context = "\n".join([doc.page_content for doc in compressed_docs])
        # Format prompt
        prompt_input = self.prompt.format(context=context, question=question)
        # Get model response
        response = self.model.invoke(prompt_input)
        answer = response.content if hasattr(response, 'content') else str(response)

        # RETURN THE ANSWER AND THE RETRIEVED DOCUMENTS USED FOR CONTEXT
        return {
            "answer": answer,
            "retrieved_docs": compressed_docs
        }

    @property
    def input_keys(self) -> List[str]:
        return ["question"]

    @property
    def output_keys(self) -> List[str]:
        return ["answer"]
