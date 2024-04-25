import sys
sys.path.append('../')
from config import config
from typing import List, Union
from langchain_openai import OpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.memory import ConversationSummaryBufferMemory


class Memory:
    def __init__(self) -> None:

        self.memory = ConversationSummaryBufferMemory(llm=OpenAI(temperature=0), 
                                                      max_token_limit=config['token_memory_limit'], 
                                                      return_messages=True,
                                                      input_key='question',
                                                      memory_key="chat_history",
                                                      output_key='answer',
                                                      )

    @staticmethod
    def _format_chat_history(chat_history: List[Union[HumanMessage, SystemMessage, AIMessage]]) -> str:
        buffer = ""
        for dialogue_turn in chat_history:
            if isinstance(dialogue_turn, HumanMessage):
                buffer += "\nHuman: " + dialogue_turn.content
                
            elif isinstance(dialogue_turn, AIMessage):
                buffer += "\nAssistant: " + dialogue_turn.content
                
            elif isinstance(dialogue_turn, SystemMessage):
                buffer += "\nSystem: " + dialogue_turn.content
                
        return buffer