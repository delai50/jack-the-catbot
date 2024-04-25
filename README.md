# Jack the CatBot
![image](https://github.com/delai50/pdfchatbot/blob/main/server/cat.jpg)

Jack the CatBot is a LLM-based chatbot designed to answer questions based on the content of PDF files. It utilizes the Gradio library for creating a user-friendly interface and LangChain for Natural Language Processing tooling.

## Features ⭐
* Process PDF files and store them in a vector database.
* Answer user queries extracting relevant information from the database and generating responses using an LLM.
* Maintain chat history and automatically summarize it when surpasses a certain number of tokens.
* Use Python code execution agent for complex queries.
* LangSmith integration.

## Prerequisites 📋
Before running the chatbot, ensure that you have the required dependencies installed. You can use:
```
conda create -n chatbot python=3.11
pip install -r requirements.txt
```

## Configuration ⚙️
The chatbot uses a configuration file (config.py) to specify the necessary paramaters. Make sure to update the configuration file with the appropriate values if you wanted to try another model, embeddings or parameters.

## Usage 📚
1. Place your keys into the .env file. 
2. Place some PDF files into the docs folder.
3. Run an instance of the app.
4. Click the "Send" button to submit your query.
5. View the chat history and responses in the interface.

## Running Locally 💻
To run the chatbot, execute the following command:

```
cd server
python mock_interface.py
```

