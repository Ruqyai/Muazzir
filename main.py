# Import necessary libraries and modules
from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DataFrameLoader
import advertools as adv
import requests
from langchain.agents import initialize_agent, AgentType
from langchain.llms import OpenAI
from langchain.agents import initialize_agent, Tool
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()
app = Flask(__name__)

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
API_KEY = os.getenv('API_KEY')
os.environ["LANGCHAIN_TRACING"] = "true"

##############################################
# Using Alriyadh Dict APIs
##############################################

# Function to search at alriyadh dictionary API
def search_at_alriyadh_dict(query):
    url = 'https://siwar.ksaa.gov.sa/api/alriyadh/search'
    headers = {
        'accept': 'application/json',
        'apikey': API_KEY
    }

    params = {'query': query}

    # Make a request to the alriyadh dictionary API
    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        # Return the senses of the word from the API response
        return response.json()[0]['senses']
    else:
        # Return an error message for invalid response type
        return {"error": "Invalid response type"}

# Initialize an OpenAI language model (llm)
llm = OpenAI(temperature=0)

# Initialize a tool for searching at alriyadh dictionary
tools = [
    Tool(
        name="search",
        func=search_at_alriyadh_dict,
        description="ملخص الرد لسؤال عن كلمة",
    )
]

# Initialize an agent with tools and llm
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

##############################################
# Crawling
##############################################

# List of URLs to crawl
crawl_urls = [
    'https://dictionary.ksaa.gov.sa/',
    'https://dictionary.ksaa.gov.sa/Search/3',
  
]

# Crawl the specified URLs and save the output to a JSON lines file
adv.crawl(crawl_urls, 'output_file.jl', follow_links=True)
crawl_df = pd.read_json('output_file.jl', lines=True)

# Extract relevant columns from the crawled data
crawl_df = crawl_df[['body_text', 'url', 'title', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']]

# Load crawled data into a DataFrameLoader
loader = DataFrameLoader(crawl_df, page_content_column="body_text")
docs = loader.load()

# Split documents into chunks for processing
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=10)
documents = text_splitter.split_documents(docs)

# Embed text using OpenAI embeddings
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents, embeddings)

# Create a retriever for document similarity search
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})

# Initialize a language model for conversation (llm)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Create a ConversationalRetrievalChain for question answering
qa = ConversationalRetrievalChain.from_llm(llm, retriever)

##############################################
# Routes
##############################################

# Route for the main index page
@app.route('/')
def index():
    return render_template('index.html')

# Route for the API connection page
@app.route('/connect_api')
def connect_api():
    return render_template('connect_api.html')

# Route for handling user chat requests
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data['message']

    chat_history = []  
    result = qa({"question": user_message, "chat_history": chat_history})
    bot_response = result["answer"]
    chat_history.append((user_message, bot_response))

    return jsonify({'response': bot_response})

# Route for handling user chat requests with agent
@app.route('/chat_agent', methods=['POST'])
def chat_agent():
    data = request.json
    user_message = data.get('message', '')
    bot_response = agent.run(user_message)
    return jsonify({'response': bot_response})

# Run the Flask application if this script is the main entry point
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port= 8080)
