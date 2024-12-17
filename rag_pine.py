####---------------------------------------------Groq---------------------------------------------------------------####
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_community.vectorstores import Pinecone as langchain_pinecone
from langchain.schema import StrOutputParser
from langchain_community.vectorstores import Qdrant
import qdrant_client
import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
import streamlit as st 



load_dotenv()

##-------------------------------------------------CLOUD----------------------------------------------------------##
# GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
# PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
# GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
# OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
# QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
# MODEL_NAME = "llama3-70b-8192"

##-------------------------------------------------LOCAL----------------------------------------------------------##
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

MODEL_NAME = "llama3-70b-8192"

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


####---------------------------------------------RAG---------------------------------------------------------------####

class RetrievalAugmentGeneration:
    def __init__(self, groq_api_key=GROQ_API_KEY, model_name=MODEL_NAME, uploaded_file_csv=None):
        self.groq_api_key = groq_api_key
        self.model_name = model_name
        self.persist_directory = 'persisted_docs'
        self.initialize_session_state()
        self.uploaded_file_csv = uploaded_file_csv
        self.loaded_doc = self.document_loader()
        self.embeddings = self.load_embeddings()
        self.vector_store = self.create_vector_store()


####---------------------------------------------SESSION STATE------------------------------------------------------####
    @staticmethod
    def initialize_session_state():
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = [] 

        if "messages" not in st.session_state:
            st.session_state.messages = []
            

####---------------------------------------DOCUMENT LOADER--------------------------------------------------####
    # @st.cache_resource
    # def document_loader(_self):
    #     data_path = 'processed_whatsapp_chat.csv'
    #     csv_loader = CSVLoader(data_path, encoding='utf-8')
    #     loaded_doc = csv_loader.load()
    #     if not loaded_doc:
    #         st.error(f"No documents loaded from {data_path}. Please check the file.")
    #     return loaded_doc
    
    
    def document_loader(self):
        if self.uploaded_file_csv:
            # Save the uploaded file temporarily
            with open("temp.csv", "wb") as f:
                f.write(self.uploaded_file_csv.getbuffer())
            file_path = "temp.csv"

            # Load the uploaded file
            csv_loader = CSVLoader(file_path, encoding='utf-8')
            loaded_doc = csv_loader.load()
            if not loaded_doc:
                st.error(f"No documents loaded from {file_path}. Please check the file.")
            return loaded_doc
        else:
            st.warning("Please upload a CSV file.")
        

####---------------------------------------CREATING VECTOR EMBEDDINGS------------------------------------------------####
    @st.cache_resource
    
    ####---------------------------------------HuggingFace Embeddings------------------------------------------------####
    def load_embeddings(_self, _embedding_type="huggingface"):
        try:
            if _embedding_type == "huggingface":
                return HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-mpnet-base-v2",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': False}
                )
            ####-----------------------------OpenAI Embeddings----------------------------------#### 
            elif _embedding_type == "openai":
                return OpenAIEmbeddings(
                    api_key=OPENAI_API_KEY,
                    model="text-embedding-ada-002",
                    max_retries=3
                )
            ####-----------------------------GoogleGenAI Embeddings----------------------------------#### 
            elif _embedding_type == "google":
                return GoogleGenerativeAIEmbeddings(
                    model='models/embedding-001',
                    task_type='retrieval_query',
                    google_api_key=GEMINI_API_KEY
                )
            else:
                raise ValueError("Unsupported embedding type selected")
        except Exception as e:
            st.error(f"Error loading embeddings: {str(e)}")
            return None
 
 ####----------------------------------------CREATING VECTOR STORES-------------------------------------------------####   
    @st.cache_resource
    
    def create_vector_store(_self, _vector_store_type="qdrant"):
        
        text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    separators=["\n\n", "\n", " ", ""]
                )
        texts = text_splitter.split_documents(_self.document_loader())
        
        try:
            ####-----------------------------PINECONE----------------------------------#### 
            if _vector_store_type == "pinecone":
                pc = Pinecone(api_key=PINECONE_API_KEY)
                index_name = "whatsapp-chats"
                index = pc.Index(index_name)
                
                # Create and return the vector store
                vector_store = PineconeVectorStore.from_documents(
                    documents=texts,
                    # documents=_self.document_loader()
                    embedding=_self.load_embeddings(),
                    index_name=index_name,
                    namespace="whatsapp_analysis"
                )
                
                return vector_store
            ####-----------------------------Quant DB----------------------------------#### 
            elif _vector_store_type == "qdrant":
                os.environ["QDRANT_HOST"]="63253eb8-6b3b-4aba-95e3-34da6a5268d7.us-east4-0.gcp.cloud.qdrant.io:6333"
                os.environ["QDRANT_COLLECTION_NAME"]="my-collection"
                
                vector_store = Qdrant(
                    client=qdrant_client.QdrantClient(
                        host=os.getenv("QDRANT_HOST"),
                        # host="localhost",
                        api_key=os.getenv("QDRANT_API_KEY"),
                        port=6333
                    ),
                    collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
                    embeddings=_self.load_embeddings()
                )
                
                return vector_store
            ####-----------------------------ChromaDB----------------------------------####
            elif _vector_store_type == "chromadb":
                vector_store = Chroma.from_documents(
                    documents=texts,
                    embedding=_self.load_embeddings(),
                    persist_directory="db/chroma"
                )
                vector_store.persist()
                
                return vector_store
            
            
            print(f"Vector store created with {len(texts)} documents")
        
        except Exception as e:
            st.error(f"Error creating vector store: {str(e)}")
            return None
    
 
 ####--------------------------------------------RETRIEVER---------------------------------------------------####   
    def retriever(self, user_query):
        print('INSIDE RETRIEVER FILE')
        
        if os.path.exists(self.persist_directory) == False:
            self.document_loader()
            self.create_vector_store()
        
        if not self.vector_store:
            return {"error": "Vector store not initialized"}
        
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        
        template_str = """You are an advanced AI assistant specializing in analyzing WhatsApp group chats. 
        You have access to a pandas dataframe containing detailed chat information. Your task is to provide 
        comprehensive insights and answer questions based on the following data columns:

        1. date: The date when each chat message was created
        2. time: The specific time when each chat message was sent
        3. member: The name of the group member or participant who sent the message
        4. message: The actual content of the message sent by a group member
        5. message_type: Indicates whether the message is 'text' or 'media'
        6. message_length: The total number of characters in the message
        7. reaction_count: The number of reactions received by each message
        8. word_count: The number of words in each message
        9. mentions: The number of times other members were tagged or mentioned in a message
        10. emojis & emoji: The number of emojis used in each message
        11. urlcount: The number of URLs or links included in each message

        When analyzing the chat data, consider the following aspects:

        - Participant Engagement:
        * Identify the most active group participants based on message frequency and length
        * Analyze patterns in participation over time (e.g., daily, weekly trends)
        * Highlight members who receive the most reactions or mentions

        - Content Analysis:
        * Determine the most frequent words or phrases used in the chat
        * Analyze the distribution of message types (text vs. media)
        * Identify trends in emoji usage and their context
        * Examine the frequency and nature of URL sharing

        - Temporal Patterns:
        * Identify the most active dates and times for chat activity
        * Analyze how chat activity varies over different time periods (e.g., weekdays vs. weekends)

        - Interaction Dynamics:
        * Analyze patterns in message reactions and their correlation with content
        * Examine how often and in what context members mention each other
        * Identify any conversation threads or topics that generate more engagement

        - Language and Communication Style:
        * Analyze the average message length and how it varies among participants
        * Identify any unique communication styles or patterns for different members

        Provide detailed, data-driven insights based on the available information. If asked about specific 
        metrics or trends, use the relevant columns to give accurate answers. If a question cannot be answered 
        with the given data, explain why and suggest what additional information might be needed.

        Be thorough in your analysis, but do not invent or assume information that is not present in the data. 
        If you're unsure about any aspect, ask for clarification or more specific queries.

        Context: {context}
        Human: {input}
        Assistant: Based on the provided context and your question, I'll analyze the WhatsApp chat data and 
        provide insights. 
        """
        
        prompt = PromptTemplate(input_variables=['context', 'input'], template=template_str)
        
        model = ChatGroq(model=self.model_name, api_key=self.groq_api_key, temperature=0)
        
        retriever_chain = (
            {"context": retriever | format_docs, "input": RunnablePassthrough()} 
            | prompt 
            | model
            | StrOutputParser()
        )
        
        response = retriever_chain.invoke(user_query)
        return {'result of search': response}
    
    # def chat(self, user_input):
    #     response = self.retriever(user_input)
    #     st.session_state.chat_history.append(('Human', user_input))
    #     st.session_state.chat_history.append(('AI', response))
        
    #     return response

