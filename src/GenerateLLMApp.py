import os
import sys
sys.path.append('../')
from config import config

# Utils modules
from index.IndexHandler import Index
from memory.MemoryHandler import Memory
from prompts.PromptFactory import PromptFactory

# LangGhain modules
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain

# LangChain agent
from langchain_experimental.utilities import PythonREPL
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_experimental.tools.python.tool import PythonREPLTool

from dotenv import load_dotenv

# Configuration
LOG_LEVEL = 'LLMAPP_BUILDER'

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com/'
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGSMITH_API_KEY")


class LLMApp(PromptFactory):
    def __init__(self, logger,):

        super().__init__()

        self.logger = logger
        self.retriever = None
                
        self.model = None
        self.memory = Memory()
        self.get_model()
        self.get_retriever()

    def get_model(self):
        self.logger.info(f"[{LOG_LEVEL}] LLM definition: ")

        self.model = ChatOpenAI(model=config["openAIModel"], 
                                temperature=config["temperature"])
    
    def get_retriever(self):
        self.logger.info(f'[{LOG_LEVEL}] Creating retriever for documents inside {config["rag_docs"]} folder.')
        
        # Create the index instance
        index = Index(config['rag_docs'], self.logger)
        
        # Load data
        data = index.load_data()

        # Split the data
        data_split = index.split_data(data)

        # Create the index
        index.create_index(data_split)

        self.retriever = index.vector_db.as_retriever(search_kwargs={"k": config["k_retrieved_chunks"]})

    def generate_qa_chain(self):
        self.logger.info(f'[{LOG_LEVEL}] Entering QA Chain...')

        QA_CHAIN_PROMPT = PromptTemplate.from_template(self.question_answer_template)

        qa_chain = ConversationalRetrievalChain.from_llm(
                                                        self.model,
                                                        chain_type = "stuff",
                                                        retriever = self.retriever,
                                                        memory=self.memory.memory,
                                                        return_source_documents=True,
                                                        combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT}
                                                        )

        self.logger.info(f'[{LOG_LEVEL}] QA Chain executed successfully...')
        
        return qa_chain
    

    def create_python_agent(self):
        self.logger.info(f'[{LOG_LEVEL}] Creating Python Agent...')

        agent = create_python_agent(
            self.model, 
            tool = PythonREPLTool(),
            verbose = True,
            handle_parsing_errors=True
        )

        return agent

    def generate_router_chain(self):
        router_prompt = PromptFactory.system_router_template

        prompt = ChatPromptTemplate.from_messages(
                        [
                            ("system", router_prompt),
                            ("human", "{question}"),
                        ]
                    )

        router = prompt | self.model | StrOutputParser()
        
        return router
    
    def genetate_neutral_chain(self):
        neutral_prompt = PromptFactory.neutral_template

        prompt = ChatPromptTemplate.from_messages(
                        [
                            ("system", neutral_prompt),
                            ("human", "{question}"),
                        ]
                    )

        neutral = prompt | self.model | StrOutputParser()

        return neutral

    def predict(self, message,):
        self.logger.info(f'[{LOG_LEVEL}] LLM chain invoke...')
        self.logger.info(f'[{LOG_LEVEL}] User question: {message}')

        router = self.generate_router_chain()
        route = router.invoke({'question': message})

        if "questionanswer" in route.lower():
            chain = self.generate_qa_chain()
            answer = chain.invoke({'question': message})

        elif "python" in route.lower():
            chain = self.create_python_agent()
            answer = chain.invoke({'input': message})
            answer["answer"] = answer["output"] 

        else:
            chain = self.genetate_neutral_chain()
            answer_ = chain.invoke({'question': message})
            answer = {"answer": answer_}

        return answer


if __name__ == '__main__':

    import logging
    import coloredlogs

    LOGGING_LEVEL = logging.INFO
    logger = logging.getLogger("chatbot_tool")
    logger.setLevel(LOGGING_LEVEL)

    coloredlogs.install(
        fmt="[%(asctime)s][%(levelname)s] %(message)s", level=LOGGING_LEVEL, logger=logger
    )
    app = LLMApp(logger)
        
    questions = [
        "What were the amazon net sales in the Q4 of 2023?",
        "What were the amazon net sales in the Q4 of 2022?",
        "Please take a look carefully and tell me what were the amazon net sales in the Q4 of 2022"
        "If the amazon net sales in the fourth quarter of 2023 were $170.0 billion and that represented a 14% increase compared to to the fourth quarter of 2022, calculate iteratively the net sales of Q4 for 2024, 2025 and 2026 and so on up to 2030",
        "Thank you!"
    ]
    
    for question in questions:
        string = f'Question: {question}'
        print(string)
        print('='*len(string))
        r = app.predict(question)
        print(r['answer'])
