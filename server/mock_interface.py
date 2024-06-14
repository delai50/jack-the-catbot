import os
import shutil
import sys
import time
import logging
import coloredlogs
import gradio as gr
import gc

sys.path.append("..")
from src.GenerateLLMApp import LLMApp
from config import config

LOGGING_LEVEL = logging.INFO
logger = logging.getLogger("chatbot_tool")
logger.setLevel(LOGGING_LEVEL)

coloredlogs.install(
    fmt="[%(asctime)s][%(levelname)s] %(message)s", level=LOGGING_LEVEL, logger=logger
)

llmap = None
def create_chatbot():
    global llmap
    llmap = LLMApp(logger)
create_chatbot()

def print_like_dislike(x: gr.LikeData):
    print(x.index, x.value, x.liked)

def add_text(history, text):
    history = history + [(text, None)]
    return history, gr.Textbox(value="", interactive=False)

def bot(history,):
    response = llmap.predict(history[-1][0])['answer']
    history[-1][1] = ""
    for character in response:
        history[-1][1] += character
        time.sleep(0.005)
        yield history

def upload_file(files):
    # Copy files to docs folder
    # It deletes the current database and creates a new one
    for file in files: shutil.copyfile(file, f"{config['rag_docs']}/{file.split('/')[-1]}")
    llmap.retriever.vectorstore._client.clear_system_cache()
    llmap.retriever.vectorstore._client = None
    shutil.rmtree(f"{config['ragdb_dir']}/rag_db_{config['embedding']}")
    create_chatbot()

def reset():
    # Reset bot and clear interface
    create_chatbot()
    initial_message = [[None, "Hey there! Jack the Catbot here! What can I help you with?"]]
    return initial_message, ""



        
with gr.Blocks(theme=gr.themes.Soft()) as demo:

    chatbot = gr.Chatbot(
        value=[[None, "Hey there! Jack the Catbot here! What can I help you with?"]],
        elem_id="chatbot",
        bubble_full_width=False,
        avatar_images=(None, (os.path.join(os.path.dirname(__file__), "cat.jpg"))),
        height=500
    )

    with gr.Row():
        txt = gr.Textbox(
            scale=4,
            show_label=False,
            placeholder="Enter text",
            container=False,
        )
        submit_button = gr.Button("🤖Submit")
        upload_button = gr.UploadButton("📁Upload", file_types=[".pdf"], file_count="multiple")
        reset_button = gr.Button("🔄Reset")

    upload_button.upload(upload_file, upload_button)

    txt_msg = submit_button.click(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
        bot, chatbot, chatbot, api_name="bot_response"
    )
    txt_msg.then(lambda: gr.Textbox(interactive=True), None, [txt], queue=False)

    reset_button.click(reset, [], [chatbot, txt])

    chatbot.like(print_like_dislike, None, None)


demo.queue()
demo.launch()
