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
from langchain_community.document_loaders import DirectoryLoader

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_openai import OpenAIEmbeddings, ChatOpenAI, OpenAI

from langchain.text_splitter import CharacterTextSplitter, Language, RecursiveCharacterTextSplitter

from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.output_parsers import PydanticOutputParser, RetryOutputParser

import time
from typing import List
from pydantic import BaseModel, Field
import os
import pandas as pd
from dotenv import load_dotenv
import json
import subprocess
import Levenshtein

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class Response(BaseModel):
    """Final response to the question being asked"""     

    answer: str = Field(description="The response to the question")
    complete_corrected_code: str = Field(description="The entire complete corrected code")
    
class Description(BaseModel):
    """Final response to the question being asked"""     

    answer: str = Field(description="The response to the question")
    

class Feedback():
        
    def __init__(self):
        pass
    
    def get_data_from_python(self):
        loader = GenericLoader.from_filesystem(
            path="./data/target",
            glob="*.py",
            parser=LanguageParser(language=Language.PYTHON, parser_threshold=500),
        )
        documents = loader.load()
        return documents
    
    def get_data_from_file(self, path):
        loader = GenericLoader.from_filesystem(
            path=path,
            glob="*.txt"
        )
        documents = loader.load()
        return documents
    
    def read_file(self, path):
        file = open(path, 'r')
        content = file.read()
        file.close()
        return content
    
    def get_python_splitter(self, docs):
        python_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON, chunk_size=2000, chunk_overlap=200
        )
        texts = python_splitter.split_documents(docs)
        return texts
    
    def get_text_splitter(self, docs):
        """ Split text """
        text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=10)
        documents = text_splitter.split_documents(docs)
        return documents
    
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
    
    def get_retriever_tool(self, retriever):
        """ Create a retriever tool """
        retriever_tool = create_retriever_tool(
            retriever,
            "retriever_tool",
            "Query a retriever to get information about user code",
        )
        return retriever_tool
    
    def parse(self, output):
        # If no function was invoked, return to user
        if "function_call" not in output.additional_kwargs:
            return AgentFinish(return_values={"output": output.content}, log=output.content)

        # Parse out the function call
        function_call = output.additional_kwargs["function_call"]
        name = function_call["name"]
        inputs = json.loads(function_call["arguments"])

        # If the Response function was invoked, return to the user with the function inputs
        if name == "Description":
            return AgentFinish(return_values=inputs, log=str(function_call))
        # Otherwise, return an agent action
        else:
            return AgentActionMessageLog(
                tool=name, tool_input=inputs, log="", message_log=[output]
            )
    
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
    
    def get_patch_agent(self, retriever_tool):
        """ Get the agent to generate a patch for the code """
        user_prompt = """
        Given the above conversation, generate a search query to look up to get information relevant to the conversation
        """
        
        system_prompt = """
        Answer the user's questions based on the below problem and code data.
        This data contains information on a programming problem and a user's Python code response to it.
        The user's code unconditionally contains an error or incorrect output.
        
        You must the find the problem of code.
        If there is an error in the code, you must identify the line of the code where the error occurs and also show it to user.
        The line starts at 1 on the whole entire file of code and increases on the basis of a new line.
        
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        The correct answer is more important than a quick answer.
        You must always think about various testcases and counterexample and the code you made should all pass this testcases.
        You must follow the output format exactly as shown below and must invoke the Response function that contains 'answer'.
        You must answer in Korean.
        You must answer it very kindly and politely.
        
        Keep the answer as concise as possible:

        Programming problem: {problem}
        User's code: {code}
        Input_testcase: {input_testcase}
        Output_testcase: {output_testcase}
        
        Response format example: 
            'answer': entire output that you made
        """
        
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo-1106", 
            # model_name="gpt-4",
            temperature=0, 
            openai_api_key=OPENAI_API_KEY,
            max_tokens=2000
        )
        
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    system_prompt
                ),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        
        llm_with_tools = llm.bind_functions([retriever_tool, Description])
        agent = (
            {
                "problem": lambda x: x["problem"],
                "code": lambda x: x["code"],
                "input_testcase": lambda x: x["input_testcase"],
                "output_testcase": lambda x: x["output_testcase"],
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_to_openai_function_messages(
                    x["intermediate_steps"]
                ),
            }
            | prompt
            | llm_with_tools
            | self.parse
        )
        
        agent_executor = AgentExecutor(tools=[retriever_tool], agent=agent, verbose=True)
        return agent_executor
    
    def get_code_fix_agent(self, retriever_tool):
        """ Get the agent to generate a code fix for the code """
        user_prompt = """
        Given the above conversation, generate a search query to look up to get information relevant to the conversation
        """
        
        system_prompt = """
        Answer the user's questions based on the below problem and code data.
        This data contains information on a programming problem and a user's Python code response to it.
        The user's code unconditionally contains an error or incorrect output.
        
        You must provide the fixed code using patch data that contains the information of error
        patch data: {patch}.
        You must provide the correct code to fix the error.
        The line starts at 1 on the whole entire file of code and increases on the basis of a new line.
        
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        The correct answer is more important than a quick answer.
        You must always think about various testcases and counterexample and the code you made should all pass this testcases.
        You must follow the output format exactly as shown below and must invoke the Response function that contains 'answer'.
        You must answer in Korean.
        You must answer it very kindly and politely.
        
        Keep the answer as concise as possible:

        Programming problem: {problem}
        User's code: {code}
        Input_testcase: {input_testcase}
        Output_testcase: {output_testcase}
        patch: {patch}
        
        Response format example: 
            'answer': entire output that you made
        """
        
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo-1106", 
            # model_name="gpt-4",
            temperature=0, 
            openai_api_key=OPENAI_API_KEY,
            max_tokens=2000
        )
        
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    system_prompt
                ),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        
        llm_with_tools = llm.bind_functions([retriever_tool, Description])
        agent = (
            {
                "problem": lambda x: x["problem"],
                "code": lambda x: x["code"],
                "input_testcase": lambda x: x["input_testcase"],
                "output_testcase": lambda x: x["output_testcase"],
                "patch": lambda x: x["patch"],
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_to_openai_function_messages(
                    x["intermediate_steps"]
                ),
            }
            | prompt
            | llm_with_tools
            | self.parse
        )
        
        agent_executor = AgentExecutor(tools=[retriever_tool], agent=agent, verbose=True)
        return agent_executor
    
    def get_correct_agent(self, retriever_tool):
        user_prompt = """
        Given the above conversation, generate a search query to look up to get information relevant to the conversation
        """
        
        system_prompt = """
        Answer the user's questions based on the below problem and code data.
        This data contains information on a programming problem and a user's Python code response to it.
        The user's code unconditionally contains an error or incorrect output.
        
        You must make the entire corrected code using patch data that show the problem of code and code_fix data that fixed patch.
        patch data: {patch}
        code_fix data: {code_fix}
        The line starts at 1 on the whole entire file of code and increases on the basis of a new line.
        If the output of the code is incorrect, you must change the code to provide the correct output.
        You must show the complete corrected code.
        
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        The correct answer is more important than a quick answer.
        The code you made must be correct and complete and must be able to run without any errors and provide the correct output. Please be careful.
        You must always think about various testcases and counterexample and the code you made should all pass this testcases.
        Entire code must be not the same as {history_code} and must be different. {history_code} is the code list that has been corrected so far.
        The answer only contains the entire corrected code.
        You must answer only the entire corrected code.
        You must answer it very kindly and politely.
        
        Keep the answer as concise as possible and the answer must only the entire corrected code:

        Programming problem: {problem}
        User's code: {code}
        Patch: {patch}
        Code fix: {code_fix}
        Input_testcase: {input_testcase}
        Output_testcase: {output_testcase}
        History code: {history_code}
        
        Response format example: 
            'answer': entire code that you made , 
        """
        
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo-1106", 
            # model_name="gpt-4",
            temperature=0, 
            openai_api_key=OPENAI_API_KEY,
            max_tokens=2000
        )
        
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    system_prompt
                ),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        
        llm_with_tools = llm.bind_functions([retriever_tool, Description])
        agent = (
            {
                "problem": lambda x: x["problem"],
                "code": lambda x: x["code"],
                "patch": lambda x: x["patch"],
                "code_fix": lambda x: x["code_fix"],
                "input_testcase": lambda x: x["input_testcase"],
                "output_testcase": lambda x: x["output_testcase"],
                "history_code": lambda x: x["history_code"],
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_to_openai_function_messages(
                    x["intermediate_steps"]
                ),
            }
            | prompt
            | llm_with_tools
            | self.parse
        )
        
        agent_executor = AgentExecutor(tools=[retriever_tool], agent=agent, verbose=True)
        return agent_executor
    
    def get_diff_agent(self, retriever_tool):
        user_prompt = """
        Given the above conversation, generate a search query to look up to get information relevant to the conversation
        """
        
        system_prompt = """
        Answer the user's questions based on the below code data.
        This data contains code that have problems and incorrect output and the code that have been corrected so far.
        
        You must find the difference between the user's code and the corrected code.
        You must make a description of the difference between the user's code and the corrected code.
        Difference can be multiple, so you must find all the differences.
        
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        The correct answer is more important than a quick answer.
        You must explain the difference between the user's code and the corrected code and the reason for the difference in a detailed manner.
        You must explain it in a very detailed and easy-to-understand manner. Not just showing the code. I don't need entire corrected code. Just show the difference.
        You must answer in Korean.
        You must answer it very kindly and politely.
        
        The output should be in the following format:
        1. The detailed description of the difference between the user's code and the corrected code. ex) user's code: triangle_base = 2 * x_right, corrected code: triangle_base = 2 * x_right + 1
        2. Explain the reason for the difference. ex) 사용자의 코드에서는 삼각형의 넓이를 계산할 때, base에 1을 더하는 부분이 빠져있습니다.
        
        Keep the answer as concise as possible:
        
        Programming problem: {problem}
        Patch: {patch}
        Code fix: {code_fix}
        User's code: {user_code}
        Corrected code: {corrected_code}
        Input_testcase: {input_testcase}
        Output_testcase: {output_testcase}
        
        Response format example: 
            'answer': entire output that you made
        """
        
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo-1106", 
            # model_name="gpt-4",
            temperature=0, 
            openai_api_key=OPENAI_API_KEY,
            max_tokens=2000
        )
        
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    system_prompt
                ),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        
        llm_with_tools = llm.bind_functions([retriever_tool, Description])
        agent = (
            {
                "problem": lambda x: x["problem"],
                "patch": lambda x: x["patch"],
                "code_fix": lambda x: x["code_fix"],
                "input_testcase": lambda x: x["input_testcase"],
                "output_testcase": lambda x: x["output_testcase"],
                "user_code": lambda x: x["user_code"],
                "corrected_code": lambda x: x["corrected_code"],
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_to_openai_function_messages(
                    x["intermediate_steps"]
                ),
            }
            | prompt
            | llm_with_tools
            | self.parse
        )
        
        agent_executor = AgentExecutor(tools=[retriever_tool], agent=agent, verbose=True)
        return agent_executor
    
    def get_recommender_agent(self, retriever_tool):
        user_prompt = """
        Given the above conversation, generate a search query to look up to get information relevant to the conversation
        """
        
        system_prompt = """
        Answer the user's questions based on the below code data.
        This data contains code that have problems and incorrect output and the code that have been corrected so far.
        
        You must find the difference between the user's code and the corrected code.
        You must make a description of the difference between the user's code and the corrected code.
        Difference can be multiple, so you must find all the differences.
        
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        The correct answer is more important than a quick answer.
        You must explain the difference between the user's code and the corrected code and the reason for the difference in a detailed manner.
        You must explain it in a very detailed and easy-to-understand manner. Not just showing the code. I don't need entire corrected code. Just show the difference.
        You must answer in Korean.
        You must answer it very kindly and politely.
        
        The output should be in the following format:
        1. The detailed description of the difference between the user's code and the corrected code. ex) user's code: triangle_base = 2 * x_right, corrected code: triangle_base = 2 * x_right + 1
        2. Explain the reason for the difference. ex) 사용자의 코드에서는 삼각형의 넓이를 계산할 때, base에 1을 더하는 부분이 빠져있습니다.
        
        Keep the answer as concise as possible:
        
        Programming problem: {problem}
        Patch: {patch}
        Code fix: {code_fix}
        User's code: {user_code}
        Corrected code: {corrected_code}
        Input_testcase: {input_testcase}
        Output_testcase: {output_testcase}
        
        Response format example: 
            'answer': entire output that you made
        """
        
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo-1106", 
            # model_name="gpt-4",
            temperature=0, 
            openai_api_key=OPENAI_API_KEY,
            max_tokens=2000
        )
        
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    system_prompt
                ),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        
        llm_with_tools = llm.bind_functions([retriever_tool, Description])
        agent = (
            {
                "problem": lambda x: x["problem"],
                "patch": lambda x: x["patch"],
                "code_fix": lambda x: x["code_fix"],
                "input_testcase": lambda x: x["input_testcase"],
                "output_testcase": lambda x: x["output_testcase"],
                "user_code": lambda x: x["user_code"],
                "corrected_code": lambda x: x["corrected_code"],
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_to_openai_function_messages(
                    x["intermediate_steps"]
                ),
            }
            | prompt
            | llm_with_tools
            | self.parse
        )
        
        agent_executor = AgentExecutor(tools=[retriever_tool], agent=agent, verbose=True)
        return agent_executor
    
    def run_python(self, code, input_testcase, output_testcase):
        for i in range(len(input_testcase.split("\n"))):
            param = input_testcase.split("\n")[i]
            # print(param)
            new_code = code + f"\nprint(solution({param}))"
            
            with open("tmp"+str(i)+".py", "w") as file:
                file.write(new_code)
                
            result = subprocess.run(["python", "tmp" + str(i) + ".py"], capture_output=True, text=True)
            if result.stderr:
                # print(result.stderr)
                return False
            result = result.stdout 
            
            # print("Result", result)
            
            if str(result).strip() != output_testcase.split("\n")[i].strip():
                # print("Result strip", str(result).strip())
                # print("Test", output_tsestcase.split("\n")[i].strip())
                return False
        return True
    
    ## 오류 부분 확인 및 수정 코드 생성 (5개)
    def make_correct_code(self, query):
        history_code = []
        
        documents = self.get_data_from_python()
        python_splitter = self.get_python_splitter(documents)
        cached_embedder = self.get_cached_embedder()
        embeddings = self.get_embeddings(python_splitter, cached_embedder)
        retriever = self.get_retriever(embeddings)
        retriever_tool = self.get_retriever_tool(retriever)
        
        problem = self.read_file('./data/problem3.txt')
        input_testcase = self.read_file('./data/input3.txt')
        output_testcase = self.read_file('./data/output3.txt')
        code = self.read_file('./data/test3.py')
        correct_agent = feedback. get_correct_agent(retriever_tool)
        patch_agent = feedback. get_patch_agent(retriever_tool)
        code_fix_agent = feedback. get_code_fix_agent(retriever_tool)
        
        patch = patch_agent(
            {
                "problem": problem,
                "code": code,
                "input_testcase": input_testcase,
                "output_testcase": output_testcase,
                "input": query,
            },
            return_only_outputs=True,
        )
        
        patch = patch['answer'] if patch.get("answer") else patch['output']
        
        print("PATCH", patch)
        
        code_fix = code_fix_agent(
            {
                "problem": problem,
                "code": code,
                "patch": patch,
                "input_testcase": input_testcase,
                "output_testcase": output_testcase,
                "input": query,
            },
            return_only_outputs=True,
        )
        
        code_fix = code_fix['answer'] if code_fix.get("answer") else code_fix['output']
        
        print("CODE FIX", code_fix)
        
        for i in range(5):
            entire_code = correct_agent(
                    {   
                        "problem": problem,
                        "code": code,
                        "patch": patch,
                        "code_fix": code_fix,
                        "input_testcase": input_testcase,
                        "output_testcase": output_testcase,
                        "history_code": history_code,
                        "input": query},
                    return_only_outputs=True)
            
            print("ENTIRE CODE", entire_code)
            
            entire_code = entire_code['answer'] if entire_code.get("answer") else entire_code['output']
            
            if feedback.run_python(entire_code, input_testcase, output_testcase):
                # return response['complete_corrected_code']
                history_code.append(entire_code)
            else:
                history_code.append('')
                
        return retriever_tool, history_code, patch, code_fix, problem, input_testcase, output_testcase
            
    ## 유사도 비교: 유사도 제일 높은 코드, 설명 반환
    def levenshtein_similarity(self, history_code, user_code):
        sim_dict = {}  
        for code in history_code:
            sim =  Levenshtein.ratio(code, user_code)
            sim_dict[code] = sim
            
        max_sim = max(sim_dict.values())
        max_code = [k for k, v in sim_dict.items() if v == max_sim]
        
        return max_code[0] 
            
    ## 줄글 설명
    def make_description(self, retriever_tool, user_code, corrected_code, patch, code_fix, problem, input_testcase, output_testcase):
        diff_agent = feedback. get_diff_agent(retriever_tool)
        
        query = "사용자의 코드와 수정된 코드의 차이점을 설명해줘. 사용자의 코드와 수정된 코드의 차이점을 설명하고, 그 이유를 설명해줘."
        
        diff = diff_agent(
            {   
                "problem": problem,
                "input_testcase": input_testcase,
                "output_testcase": output_testcase,
                "user_code": user_code,
                "corrected_code": corrected_code,
                "patch": patch,
                "code_fix": code_fix,
                "input": query},
            return_only_outputs=True)
        
        diff = diff['answer'] if diff.get("answer") else diff['output']
        
        print("DIFF", diff)
        
        return diff
    
    def make_recommender(self, retriever_tool, user_code, corrected_code, patch, code_fix, problem, input_testcase, output_testcase):
        recommender_agent = feedback. get_recommender_agent(retriever_tool)
        
        query = "사용자에게 추천하는 프로그래밍 문제 유형, 예시, 강의 등을 추천해줘"
        
        recommender = recommender_agent(
            {   
                "problem": problem,
                "input_testcase": input_testcase,
                "output_testcase": output_testcase,
                "user_code": user_code,
                "corrected_code": corrected_code,
                "patch": patch,
                "code_fix": code_fix,
                "input": query},
            return_only_outputs=True)
        
        recommender = recommender['answer'] if recommender.get("answer") else recommender['output']
        
        print("RECOMMENDER", recommender)
        
        return recommender
        

if __name__ == "__main__":
    feedback = Feedback()
    
    query = "이 파이썬 코드의 문제점을 진단해줘. 에러가 발생하는 부분의 코드를 보여주고 올바른 코드로 수정해줘. 그리고 완성된 전체 코드를 보여줘 "
    
    user_code = feedback.read_file('./data/target/test.py')
    
    ## 1. 오류 부분 확인(패치 -> 수정 -> 전체 코드 생성)
    retriever_tool, history_code, patch, code_fix, problem, input_testcase, output_testcase = feedback.make_correct_code(query)
    
    ## 2. 유사도 비교: 유사도 제일 높은 코드, 설명 반환
    corrected_code = feedback.levenshtein_similarity(history_code, user_code)
    
    ## 3. 줄글 설명
    result = feedback.make_description(retriever_tool, user_code, corrected_code, patch, code_fix, problem, input_testcase, output_testcase)
    
    ## 4. 추천
    recommender = feedback.make_recommender(retriever_tool, user_code, corrected_code, patch, code_fix, problem, input_testcase, output_testcase)
    # print("RESULT", result) 

