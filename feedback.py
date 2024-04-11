from langchain.storage import InMemoryStore, LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain.tools.retriever import create_retriever_tool

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.agents import AgentActionMessageLog, AgentFinish
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate

from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_function_messages

from langchain_community.document_transformers import  EmbeddingsRedundantFilter
from langchain_community.vectorstores import Chroma,  Qdrant, FAISS
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_openai import OpenAIEmbeddings, ChatOpenAI, OpenAI

from langchain.text_splitter import CharacterTextSplitter, Language, RecursiveCharacterTextSplitter

from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline

import time
from typing import List
from pydantic import BaseModel, Field
import os
import pandas as pd
from dotenv import load_dotenv
import json

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY_KSW")

class Response(BaseModel):
    """Final response to the question being asked"""

    answer: str = Field(description="The final answer to respond to the user")
    sources: List[int] = Field(
        description="List of page chunks that contain answer to the question. Only include a page chunk if it contains relevant information"
)

class Feedback():
        
    def __init__(self):
        pass
    
    def get_data_from_python(self):
        loader = GenericLoader.from_filesystem(
            path="./data",
            glob="*.py",
            parser=LanguageParser(language=Language.PYTHON, parser_threshold=500),
        )
        documents = loader.load()
        return documents
    
    def get_python_splitter(self, docs):
        python_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON, chunk_size=2000, chunk_overlap=200
        )
        texts = python_splitter.split_documents(docs)
        return texts
    
    # def get_text_splitter(self, docs):
    #     """ Split text """
    #     text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=10)
    #     documents = text_splitter.split_documents(docs)
    #     return documents
    
    def get_cached_embedder(self):
        """ Get cached embedder -> Speed up """""
        underlying_embeddings = OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY)
        store = LocalFileStore("./cache/")
        cached_embedder = CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings, store, namespace=underlying_embeddings.model
        )
        return cached_embedder
    
    def get_embeddings(self, documents, cached_embedder, collection_name="code"):
        vectorstore = Chroma.from_documents(
            documents, 
            cached_embedder,
            collection_name=collection_name,
        )
        return vectorstore
    
    def get_retriever(self, vectorstore):
        """ Create a retriever """
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 8},
        )
        return retriever
    
    def get_pipeline_compression_retriever(self, retriever, embeddings):
        """ Create a pipeline of document transformers and a retriever """
        ## filters
        splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=20)
        redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
        # threshold
        relevant_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76) 
        
        pipeline_compressor = DocumentCompressorPipeline(
            transformers=[splitter, redundant_filter, relevant_filter]
        )
        
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=pipeline_compressor, base_retriever=retriever
        )
        return compression_retriever
            
    def get_agent(self, retriever):
        
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo-1106", 
            # model_name="gpt-4",
            temperature=0, 
            openai_api_key=OPENAI_API_KEY,
            max_tokens=2000
        )
        
        prompt = ChatPromptTemplate.from_messages(
            [
                ("user", "{input}"),
                (
                    "user",
                    "Given the above conversation, generate a search query to look up to get information relevant to the conversation",
                ),
            ]
        )

        retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Answer the user's questions based on the below context:\n\n{context}",
                ),
                ("user", "{input}"),
            ]
        )
        
        document_chain = create_stuff_documents_chain(llm, prompt)
        qa = create_retrieval_chain(retriever_chain, document_chain)
        
        return qa

if __name__ == "__main__":
    feedback = Feedback()
    
    documents = feedback.get_data_from_python()
    python_splitter = feedback.get_python_splitter(documents)
    cached_embedder = feedback.get_cached_embedder()
    embeddings = feedback.get_embeddings(python_splitter, cached_embedder)
    retriever = feedback.get_retriever(embeddings)
    # compression_retriever = feedback.get_pipeline_compression_retriever(retriever, embeddings)
    agent = feedback.get_agent(retriever)
    
    query = "What is a RunnableBinding?"
    result = agent.invoke({"input": query})
    print(result["answer"])

