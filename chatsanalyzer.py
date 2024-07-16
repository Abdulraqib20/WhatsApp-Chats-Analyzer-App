# Import libraries and packages
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import base64
import os
import re
import emoji
import warnings;warnings.filterwarnings(action='ignore')


# Create a Streamlit app
st.set_page_config(
    page_title="Raqib's WhatsApp Chats Analyzer",
    page_icon="icons8-whatsapp-48.png",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Styling ---
st.markdown(
    """
    <style>
        /* General */
        body {
            font-family: 'Arial', sans-serif;
            color: #333;
            background-color: #f4f4f9;
        }

        /* Header */
        .main-header {
            color: white;
            padding: 20px;
            text-align: center;
            border-radius: 10px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }

        .main-header h1 {
            font-size: 2.5rem;
            margin: 0;
        }

        /* Rocket emoji animation */
        .main-header h1 span {
            display: inline-block;
            animation: rocket-animation 2s linear infinite;
        }

        @keyframes rocket-animation {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-10px); }
        }

        /* WhatsApp icon */
        .whatsapp-icon {
            height: 50px;
            margin-right: 15px;
            vertical-align: middle;
        }

        /* Intro and get started sections */
        .section {
            # background-color: #fff;
            border: 1px solid #ddd;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }

        .section h2, .section h3 {
            color: #007bff;
            margin-bottom: 15px;
        }

        .section p {
            line-height: 1.6;
        }

        /* Expander header */
        .stExpanderHeader {
            color: white;
            padding: 10px 15px;
            font-weight: bold;
            border-radius: 5px;
            cursor: pointer;
        }

        .stExpanderContent p {
            margin-top: 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Header ---
with open("icons8-whatsapp-48.png", "rb") as f:  
    favicon_data = f.read()
    b64_favicon = base64.b64encode(favicon_data).decode()

st.markdown(
    f"""
    <div class="main-header">
        <h1><img src="data:image/png;base64,{b64_favicon}" class="whatsapp-icon">WhatsApp Chats Analyzer</h1> 
    </div>
    """,
    unsafe_allow_html=True,
)




# Create tabs for different sections
# Custom HTML, CSS, and JavaScript for animated tabs
custom_html = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&display=swap');

.tabs-container {
    font-family: 'Poppins', sans-serif;
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    border-radius: 20px;
    box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
}

.tab-buttons {
    display: flex;
    justify-content: center;
    margin-bottom: 20px;
}

.tab-button {
    background: none;
    border: none;
    padding: 10px 20px;
    font-size: 18px;
    font-weight: 500;
    color: #333;
    cursor: pointer;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.tab-button::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 0;
    width: 100%;
    height: 3px;
    background-color: #3498db;
    transform: scaleX(0);
    transition: transform 0.3s ease;
}

.tab-button:hover::after,
.tab-button.active::after {
    transform: scaleX(1);
}

.tab-content {
    background: rgba(255, 255, 255, 0.8);
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    transition: all 0.5s ease;
    opacity: 0;
    transform: translateY(20px);
}

.tab-content.active {
    opacity: 1;
    transform: translateY(0);
}

.feature-list {
    list-style-type: none;
    padding: 0;
}

.feature-list li {
    margin-bottom: 10px;
    padding-left: 25px;
    position: relative;
}

.feature-list li::before {
    content: '🚀';
    position: absolute;
    left: 0;
    top: 0;
}

</style>

<div class="tabs-container">
    <div class="tab-buttons">
        <button class="tab-button active" onclick="showTab('how-to-use')">How To Use</button>
        <button class="tab-button" onclick="showTab('about')">About</button>
    </div>
    
    <div id="how-to-use" class="tab-content active">
        <h2>How To Use</h2>
        <p>
            Welcome to our WhatsApp Chat Analyzer, an exciting journey through your messages! Here's your guide to unlocking insights:
        </p>
        <ol>
            <li><strong>Export Your Chat:</strong> In WhatsApp, select the chat you want to analyze and export it (without media).</li>
            <li><strong>Upload:</strong> Use the sidebar uploader to bring your chat file into our magical analyzer.</li>
            <li><strong>Explore:</strong> Dive into various tabs filled with fascinating analyses and visualizations.</li>
            <li><strong>Interact:</strong> Many charts are your playground - hover, zoom, and discover hidden stories!</li>
            <li><strong>AI Chat:</strong> Use our Generative AI feature to have a conversation with your own chat history!</li>
            <li><strong>Share:</strong> Found a gem? Share your discoveries with friends and spark interesting conversations!</li>
        </ol>
    </div>
    
    <div id="about" class="tab-content">
        <h2>About</h2>
        <p>
            Embark on a data-driven adventure with our WhatsApp Chat Analyzer! Uncover the hidden patterns in your conversations and gain fascinating insights.
        </p>
        <h3>✨ Magical Features:</h3>
        <ul class="feature-list">
            <li>Analyze individual chats or group dynamics</li>
            <li>Visualize message frequency with stunning charts</li>
            <li>Identify chat champions in your groups</li>
            <li>Discover prime-time chatting hours</li>
            <li>Explore your vocabulary and emoji game</li>
            <li>Chat with AI about your own conversations!</li>
        </ul>
        <p>
            Your privacy is our top priority. No data is stored or shared. It's just you and your insights!
        </p>
        <p>
            Ready to unlock the secrets of your chats? Let's dive in and let the fun begin! 🎉
        </p>
    </div>
</div>

<script>
function showTab(tabId) {
    // Hide all tab contents
    var tabContents = document.getElementsByClassName('tab-content');
    for (var i = 0; i < tabContents.length; i++) {
        tabContents[i].classList.remove('active');
    }
    
    // Show the selected tab content
    document.getElementById(tabId).classList.add('active');
    
    // Update active state of tab buttons
    var tabButtons = document.getElementsByClassName('tab-button');
    for (var i = 0; i < tabButtons.length; i++) {
        tabButtons[i].classList.remove('active');
    }
    event.currentTarget.classList.add('active');
}
</script>
"""

# Render the custom HTML
components.html(custom_html, height=600)




# --- Display a GIF image with a caption ---
st.title(" ")
st.caption("Demo on how to export WhatsApp chats to a Text (.txt) File")
video_url = "demo.gif" 
st.image(video_url)


def date_time(s):
    pattern = r'^\d{1,2}/\d{1,2}/\d{2}, \d{1,2}:\d{2}'
    result = re.match(pattern, s)
    if result:
        return True
    return False

def data_point(line):
    split_line = line.split(' - ')
    date_time_str = split_line[0]
    date_str, time_str = date_time_str.split(', ')
    message = ' - '.join(split_line[1:])
    
    return date_str, time_str, message

st.title(" ")
# Create a file upload widget
uploaded_file = st.sidebar.file_uploader("Upload your WhatsApp chats .txt file", type=["txt"])

try:
    

    # Check if a file was uploaded
    if uploaded_file:
        st.success("File successfully uploaded.")

        # Process the uploaded file and convert to a Pandas DataFrame
        with st.spinner("Loading📍..."):
            chat_data = uploaded_file.read().decode('utf-8').splitlines()

            data = []

            message_buffer = []
            date, time, author = None, None, None
            for line in chat_data:
                line = line.strip()
                if date_time(line):
                    if len(message_buffer) > 0:
                        for message in message_buffer:
                            data.append([date, time, author, message])
                        message_buffer.clear()
                    date, time, message = data_point(line)
                    author, message = message.split(': ', 1) if ': ' in message else (np.nan, message)
                    message_buffer.append(message)
                else:
                    message_buffer.append(line)

            # Create a DataFrame
            df = pd.DataFrame(data, columns=['date', 'time', 'member', 'message'])

            df['message_type'] = df['message'].apply(lambda x: 'media' if 'attached' in x else 'text')
            df['message_length'] = df['message'].apply(len)
            df['reaction_count'] = df['message'].apply(lambda x: len(re.findall(r'👍|❤️|😂|😢|😮|😡|🎉|👏', x)))
            df['word_count'] = df['message'].apply(lambda x: len(re.findall(r'\b\w+\b', x)))
            df['mentions'] = df['message'].apply(lambda x: ", ".join(re.findall(r'@(\w+)', x)))
            # df['message'] = df.apply(lambda row: row['message'].replace(f"{row['member']}: ", ""), axis=1)
            # df['message'] = df['message'].str.replace(f"{df['member']}: ", "", regex=True)
            # df['member'] = df['member'].apply(lambda x: re.escape(x))
            # df['message'] = df['message'].str.replace(f"{df['member']}: ", "", regex=True)

            # Function to remove member names from messages
            def remove_member_names(df):
                for index, row in df.iterrows():
                    message = str(row['message'])  # Ensure 'message' is a string
                    member = str(row['member'])    # Ensure 'member' is a string
                    if message.startswith(member + ': '):
                        df.at[index, 'message'] = message[len(member) + 2:]
                return df

            # Apply the function to remove member names
            df = remove_member_names(df)

            # get emojis
            def extract_emojis(message):
                return [emoji.emojize(c) for c in message if emoji.is_emoji(c)]

            df['emojis'] = df['message'].apply(extract_emojis)

            # Remove non-breaking space from the "time" column
            # df['time'] = df['time'].str.replace('\u202F', ' ')
            # df['time'] = pd.to_datetime(df['time'], format='%I:%M %p').dt.strftime('%I:%M %p')

            # Remove non-breaking space from the "time" column
            df['time'] = df['time'].str.replace('\u202F', ' ')
            if df['time'].str.match(r'^\d{2}:\d{2}$').all():
                time_format = '%H:%M'  # 24-hour format
            else:
                time_format = '%I:%M %p'  # AM/PM format

            df['time'] = pd.to_datetime(df['time'], format=time_format).dt.strftime('%I:%M %p')
            df['date'] = pd.to_datetime(df['date'], format='%m/%d/%y')
            df = df.dropna(subset=['member'])

            st.markdown(
                f"""
                <style>
                    div.stButton > button:first-child {{
                        background-color: #636EFA;
                        color: white;
                        font-weight: bold;
                        font-size: 18px;
                    }}
                </style>
                """,
                unsafe_allow_html=True,
            )

            # Create a checkbox to toggle the visibility of the DataFrame
            show_data = st.checkbox("Display Data")

            # Display the DataFrame if the checkbox is checked
            if show_data:
                st.dataframe(df)

                # Create a multiselect widget to select columns for unique values
                selected_columns = st.multiselect("Display Unique Values of Columns", df.columns.tolist())

                # Create an expander for displaying unique values
                with st.expander("Unique Values", expanded=False):
                    for column in selected_columns:
                        st.subheader(f"Unique Values in {column}")
                        unique_values = pd.unique(df[column])
                        # Use custom CSS to enable scrolling
                        st.markdown(
                            f'<div style="max-height: 300px; overflow-y: scroll;">{pd.DataFrame(unique_values, columns=[column]).to_html(index=False)}</div>',
                            unsafe_allow_html=True
                        )
            def convert_df_to_csv(df):
                return df.to_csv(index=False).encode('utf-8')
            
            if st.button("Export Data to CSV File"):
                csv = convert_df_to_csv(df)
                
                # Get the original filename without extension
                original_filename = os.path.splitext(uploaded_file.name)[0]
                
                # Set the CSV filename
                csv_filename = f"{original_filename}.csv"
                
                st.write(f"Exporting to {csv_filename}...")
                st.download_button(
                    label="Download data as CSV", 
                    data=csv, 
                    file_name=csv_filename, 
                    key="csv_download", 
                    mime='text/csv',
                )
       
            
    else:
        st.warning("Please upload a WhatsApp chat .txt file.")

    st.markdown("<br>", unsafe_allow_html=True) # line spacing
    # keyword search
    st.subheader("Keyword Search")

    # Create a text input widget for users to enter keywords
    search_keyword = st.text_input("Enter Keyword")

    # Create a button with custom CSS
    search_button = st.button("Search", key="search_button")

    # Filter the DataFrame based on the entered keyword when the button is clicked
    if search_button:
        if search_keyword:
            filtered_df = df[df['message'].str.contains(search_keyword, case=False, na=False)]

            # Display the filtered results in a table
            if not filtered_df.empty:
                st.write("Search Results:")
                st.dataframe(filtered_df)
            else:
                st.warning("No matching messages found.")
        else:
            st.warning("Please enter a keyword to search.")            

    st.markdown("<br>", unsafe_allow_html=True) # line spacing
    # Quick Stats
    st.title("Stats")

    st.subheader("Overall Stats")

    def split_count(text):

        emoji_list = []
        data = re.findall(r'[^\s\u1f300-\u1f5ff]', text)
        for word in data:
            if any(char in emoji.distinct_emoji_list(text) for char in word):
                emoji_list.append(word)

        return emoji_list

    # Calculate total messages
    total_messages = df.shape[0]
    # avg messages
    avg_message_length = df['message_length'].mean()

    # Calculate media messages
    media_messages = df[df['message'] == '<Media omitted>'].shape[0]

    # Calculate emojis
    df["emoji"] = df["message"].apply(split_count)
    emojis = sum(df['emoji'].str.len())

    # Calculate URL links
    URLPATTERN = r'(https?://\S+)'
    df['urlcount'] = df.message.apply(lambda x: re.findall(URLPATTERN, x)).str.len()
    links = np.sum(df.urlcount)

    # Display quick stats
    st.write(f"Total Messages: {total_messages}")
    st.write(f'Average message length: {avg_message_length}')
    st.write(f"Media Messages: {media_messages}")
    st.write(f"Total Emojis: {emojis}")
    st.write(f"Total Links: {links}")

    st.subheader("Member Stats")

    # Get unique member names
    unique_members = df['member'].unique()

    # Create a dropdown to select a member
    selected_member = st.selectbox("Select a Group Participant", unique_members)

    # Display individual member stats when a member is selected
    if selected_member:
        st.subheader(f"Stats for Partcipant {selected_member}")
        member_df = df[df['member'] == selected_member]

        # Calculate Messages Sent
        messages_sent = member_df.shape[0]

        # Calculate Percentage of Messages Sent out of Total
        total_messages = df.shape[0]
        percentage_messages_sent = (messages_sent / total_messages) * 100

        # Calculate Words per message
        words_per_message = member_df['word_count'].mean()

        # Calculate Media Messages Sent
        media_messages_sent = member_df[member_df['message'] == '<Media omitted>'].shape[0]

        # Calculate Emojis Sent
        member_df["emoji"] = member_df["message"].apply(split_count)
        emojis_sent = sum(member_df['emoji'].str.len())

        # Calculate Links Sent
        URLPATTERN = r'(https?://\S+)'
        member_df['urlcount'] = member_df.message.apply(lambda x: re.findall(URLPATTERN, x)).str.len()
        links_sent = np.sum(member_df.urlcount)

        # Display individual member stats
        st.write(f"Messages Sent: {messages_sent} ({percentage_messages_sent:.2f}% of total messages)")
        st.write(f"Words per Message: {words_per_message}")
        st.write(f"Media Messages Sent: {media_messages_sent}")
        st.write(f"Emojis Sent: {emojis_sent}")
        st.write(f"Links Sent: {links_sent}")



###--------------------------------------------------------------------------------------------------------------------###
###--------------------------------------------------------------------------------------------------------------------###
###--------------------------------------------------------------------------------------------------------------------###
###--------------------------------------------------------------------------------------------------------------------###
###--------------------------------------------------------------------------------------------------------------------###
###--------------------------------------------------------------------------------------------------------------------###
###--------------------------------------------------------------------------------------------------------------------###
###--------------------------------------------------------------------------------------------------------------------###
    

###----------------------------------------CHATTING & VISUALIZATIONS--------------------------------------------------------------####

    #----------------------------------------------STYLING TABS---------------------------------------------------------#

    # Title for the tabs section
    st.markdown("<h2 style='text-align: center; margin-bottom: 10px;'>Additional Features</h2>", 
                unsafe_allow_html=True)

    # Tabs for different sections of the app
    sec1, sec2 = st.tabs(["📊 Visualizations", "💬 ChatGPT"])
    
#-----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------VIZUALIZATION SECTION-----------------------------------------------------#
    
    
    with sec1:
        
        st.markdown("""
            <style>
                .stTabs {
                    overflow-x: auto;
                }
                .stTabs [data-baseweb="tab-list"] {
                    display: flex !important;
                    flex-wrap: nowrap !important;
                    overflow-x: auto !important;
                    white-space: nowrap !important;
                    border-bottom: none !important;
                    -webkit-overflow-scrolling: touch !important;
                    background-color: #075E54 !important; /* WhatsApp dark green */
                }
                .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {
                    display: none !important;
                }
                .stTabs [data-baseweb="tab-list"] {
                    -ms-overflow-style: none !important;
                    scrollbar-width: none !important;
                }
                .stTabs [data-baseweb="tab"] {
                    flex: 0 0 auto !important;
                    padding: 10px 20px !important;
                    font-size: 16px !important;
                    cursor: pointer !important;
                    background-color: #075E54 !important; /* WhatsApp dark green */
                    color: #ffffff !important;
                    border: none !important;
                    transition: background-color 0.3s ease, color 0.3s ease !important;
                    margin-right: 5px !important;
                }
                .stTabs [data-baseweb="tab"]:hover {
                    background-color: #128C7E !important; /* WhatsApp light green */
                }
                .stTabs [aria-selected="true"] {
                    color: #075E54 !important; /* WhatsApp dark green */
                    background-color: #ffffff !important;
                    border-top-left-radius: 5px !important;
                    border-top-right-radius: 5px !important;
                }
                .stTabs [data-baseweb="tab-panel"] {
                    padding: 20px !important;
                }
            </style>
            """, unsafe_allow_html=True)   

        from viz import show_viz
        show_viz(df)


#-----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------CHATGPT SECTION----------------------------------------------------------#
    
    
    with sec2:
        
    ##----------------------------------Import Libraries-------------------------------------------------------##
        st.header('Chat with the WhatsApp Analyzer 🤖')
        
        import json
        from langchain_core.prompts import (
        HumanMessagePromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate
        )
        from langchain_core.messages import HumanMessage
        from langchain_core.exceptions import LangChainException
        from langchain_core.prompts import ChatPromptTemplate
        from langchain.pydantic_v1 import BaseModel, Field
        from langchain_core.prompts import (
        HumanMessagePromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate
        )
        from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
        from langchain_groq.chat_models import ChatGroq
        from langchain_core.exceptions import LangChainException
        from langchain.tools import StructuredTool
        from langchain_core.utils.function_calling import convert_to_openai_function
        import os
        from io import BytesIO
        from rag_pine import RetrievalAugmentGeneration
        
        from dotenv import load_dotenv
        load_dotenv()
        GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
        PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
        GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
        OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
        MODEL_NAME = "llama3-70b-8192"

        uploaded_file_csv = st.sidebar.file_uploader("Upload your processed WhatsApp chat CSV file", 
                                            type=['csv'], 
                                            key="csv_uploader")

        if uploaded_file_csv is not None:
            rag_class = RetrievalAugmentGeneration(
                groq_api_key=GROQ_API_KEY,
                model_name=MODEL_NAME,
                uploaded_file_csv=uploaded_file_csv
                
            )
        else:
            st.warning("Please upload a CSV file to proceed with the analysis.")
            rag_class = None 


        



        # def process_chat(model, user_input, chat_history):
        #     return model.invoke({
        #     'chat_history' : chat_history,
        #     'input' : user_input
        #     }
        #     )

        def process_chat(model,user_input,chat_history):

            response = model.invoke(
                {
                    'chat_history' : chat_history,
                    'input' : user_input
                }
            )
            return response



#---------------------------------------------------------------------------------------------------------------------# 

        # def process_chat(model, user_input, chat_history):
        #     response = model.invoke({
        #         'chat_history': chat_history,
        #         'input': user_input
        #     })
            
        #     if response.content:
        #         return response
        #     elif response.additional_kwargs.get('tool_calls'):
        #         for tool_call in response.additional_kwargs['tool_calls']:
        #             if tool_call['function']['name'] == 'WhatsAppChatsAnalysis':
        #                 query = json.loads(tool_call['function']['arguments'])['query']
        #                 analysis_result = whatsapp_chats_analysis(query)
                        
        #                 # Generate a response based on the analysis result
        #                 follow_up_response = model.invoke({
        #                     'chat_history': chat_history,
        #                     'input': f"Based on the analysis of the WhatsApp chat data for the query '{query}', here's what I found: {analysis_result}\nPlease provide insights based on this information."
        #                 })
                        
        #                 return follow_up_response
            
        #     return AIMessage(content="I'm sorry, I couldn't process that request.")
            
        # # Define input schema for WhatsApp chats analysis
        # class WhatsAppChatsInput(BaseModel):
        #     """
        #     Input schema for analyzing WhatsApp chats from a CSV file.
        #     """
        #     query: str = Field(..., description='Query about the WhatsApp Chats')

        # # Create the function for RAG
        # def whatsapp_chats_analysis(query: str) -> str:
        #     """Analyze WhatsApp chats based on the given query."""
        #     return rag_class.retriever(query)

        # # create Structured Tool for function calling
        # whatsapp_tool = StructuredTool.from_function(
        #     func=whatsapp_chats_analysis,
        #     name="WhatsAppChatsAnalysis",
        #     description="Analyze Whatsapp chats based on the given query"
        #     )
        
        # #  CREATE CHAT TEMPLATE
        # system_prompt = """
        #     You are an advanced AI assistant developed by raqibcodes specializing in analyzing WhatsApp group chats. 
        #     Your primary role is to provide insightful analysis and answer inquiries based on group conversations. 
        #     Your responses should be thorough, detailed, and data-driven.

        #     Key Responsibilities:
        #     1. Analyze chat data comprehensively, focusing on:
        #     - Participant engagement and activity levels
        #     - Content patterns and trends
        #     - Temporal dynamics of conversations
        #     - Interaction patterns among group members
        #     - Language use and communication styles

        #     2. Utilize the following data columns in your analysis:
        #     - date: Message creation date
        #     - time: Message creation time
        #     - member: Name of the message sender
        #     - message: Actual message content
        #     - message_type: 'text' or 'media'
        #     - message_length: Character count of the message
        #     - reaction_count: Number of reactions to the message
        #     - word_count: Number of words in the message
        #     - mentions: Count of member mentions in the message
        #     - emojis & emoji: Number of emojis in the message
        #     - urlcount: Number of URLs in the message

        #     3. Maintain accuracy and precision:
        #     - Use exact names and details as they appear in the chats' member column
        #     - Do not alter or omit information, including brackets and asterisks
        #     - Example: "Raqib Omotosho: Hello everyone!" should be quoted exactly

        #     4. Provide clear, informative responses:
        #     - Offer statistical insights when relevant
        #     - Explain trends and patterns observed in the data
        #     - Highlight notable interactions or conversation dynamics

        #     5. Leverage available tools:
        #     - Utilize any provided analysis tools to enhance your insights
        #     - Clearly indicate when you're using specific tools or functions

        #     6. Handle uncertainties appropriately:
        #     - If information is unclear or insufficient, ask for clarification
        #     - Avoid making assumptions; stick to available data

        #     7. Adapt to user needs:
        #     - Tailor your analysis to the specific questions or areas of interest expressed by the user
        #     - Offer additional relevant insights beyond the direct question when appropriate

        #     Remember, your goal is to provide valuable, accurate, and actionable insights from the WhatsApp chat data, 
        #     enhancing the user's understanding of the group's communication dynamics.
        #     """

        # prompt = ChatPromptTemplate.from_messages([
        #         SystemMessagePromptTemplate.from_template(system_prompt),
        #         MessagesPlaceholder(variable_name='chat_history'),
        #         HumanMessagePromptTemplate.from_template("{input}")
        # ])

        # llm = ChatGroq(
        #     groq_api_key=GROQ_API_KEY, 
        #     model_name=MODEL_NAME, 
        #     temperature=0,
        #     max_retries=5
        # )

        # # Convert to function declaration object
        # # whatsapp_chats_func = convert_to_openai_function(WhatsAppChatsInput)
        # chat_with_tools = llm.bind_tools(tools=[whatsapp_tool])
        # chain = prompt | chat_with_tools
            
        # user_input = st.chat_input('How can I help you with analyzing your WhatsApp Chats?', key='User_input')
        
        # if user_input:
        #     st.session_state.messages.append({"role": "user", "content": user_input})
        #     with st.chat_message("user", avatar="🧑‍💻"):
        #         st.markdown(user_input)

        #     with st.chat_message('assistant', avatar="🤖"):
        #         response = process_chat(chain, user_input, st.session_state.chat_history)
                
        #         if user_input:
        #             st.session_state.chat_history.append(HumanMessage(content=user_input))
                
        #         if isinstance(response, (AIMessage, HumanMessage)):
        #             st.markdown(response.content)
        #             st.session_state.messages.append({"role": "assistant", "content": response.content})
        #             st.session_state.chat_history.append(response)
        #         else:
        #             st.error("Unexpected response format from the AI.")
        
        
#---------------------------------------------------------------------------------------------------------------------#



      
        
        # Define input schema for WhatsApp chats analysis
        class WhatsAppChatsInput(BaseModel):
            """
            Input schema for analyzing WhatsApp chats from a CSV file.
            """
            query: str = Field(..., description='Query about the WhatsApp Chats')
        
        #/// CONVERTING TO FUNCTION DECLARATION OBJECT
        whatsapp_chats_func = convert_to_openai_function(WhatsAppChatsInput)
        
        

        # # Create the function for RAG
        # def whatsapp_chats_analysis(query: str) -> str:
        #     """Analyze WhatsApp chats based on the given query."""
        #     return rag_class.retriever(query)

        # # create Structured Tool for function calling
        # whatsapp_tool = StructuredTool.from_function(
        #     func=whatsapp_chats_analysis,
        #     name="WhatsAppChatsAnalysis",
        #     description="Analyze Whatsapp chats based on the given query"
        #     )

        #  CREATE CHAT TEMPLATE
        system_prompt = """
            You are an advanced AI assistant developed by raqibcodes specializing in analyzing WhatsApp group chats. 
            Your primary role is to provide insightful analysis and answer inquiries based on group conversations. 
            Your responses should be thorough, detailed, and data-driven.

            Key Responsibilities:
            1. Analyze chat data comprehensively, focusing on:
            - Participant engagement and activity levels
            - Content patterns and trends
            - Temporal dynamics of conversations
            - Interaction patterns among group members
            - Language use and communication styles

            2. Utilize the following data columns in your analysis:
            - date: Message creation date
            - time: Message creation time
            - member: Name of the message sender
            - message: Actual message content
            - message_type: 'text' or 'media'
            - message_length: Character count of the message
            - reaction_count: Number of reactions to the message
            - word_count: Number of words in the message
            - mentions: Count of member mentions in the message
            - emojis & emoji: Number of emojis in the message
            - urlcount: Number of URLs in the message

            3. Maintain accuracy and precision:
            - Use exact names and details as they appear in the chats' member column
            - Do not alter or omit information, including brackets and asterisks
            - Example: "Raqib Omotosho: Hello everyone!" should be quoted exactly

            4. Provide clear, informative responses:
            - Offer statistical insights when relevant
            - Explain trends and patterns observed in the data
            - Highlight notable interactions or conversation dynamics

            5. Leverage available tools:
            - Utilize any provided analysis tools to enhance your insights
            - Clearly indicate when you're using specific tools or functions

            6. Handle uncertainties appropriately:
            - If information is unclear or insufficient, ask for clarification
            - Avoid making assumptions; stick to available data

            7. Adapt to user needs:
            - Tailor your analysis to the specific questions or areas of interest expressed by the user
            - Offer additional relevant insights beyond the direct question when appropriate

            Remember, your goal is to provide valuable, accurate, and actionable insights from the WhatsApp chat data, 
            enhancing the user's understanding of the group's communication dynamics.
            """

        # prompt = ChatPromptTemplate.from_messages([
        #         SystemMessagePromptTemplate.from_template(system_prompt),
        #         MessagesPlaceholder(variable_name='chat_history'),
        #         HumanMessagePromptTemplate.from_template("{input}")
        # ])
        
        prompt = ChatPromptTemplate.from_messages([
            ('system' , system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template('{input}')
        ])

        chat_model = ChatGroq(
            groq_api_key=GROQ_API_KEY, 
            model_name=MODEL_NAME, 
            temperature=0,
            max_retries=5
        )
        
        chat_with_tools = chat_model.bind_tools(tools=[whatsapp_chats_func]) 
        chain =  prompt | chat_with_tools
        
        
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = [] 

        if "messages" not in st.session_state:
            st.session_state.messages = []


        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar="🧑‍💻" if message["role"] == "user" else "🤖"):
                st.markdown(message["content"])

        if user_input := st.chat_input('How can I help you with analyzing your WhatsApp Chats?', key='user_input'):
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(user_input)

            with st.chat_message('assistant', avatar="🤖"):

                response = process_chat(chain,user_input,st.session_state.chat_history)
                st.session_state.chat_history.append(HumanMessage(content=user_input))
                st.session_state.chat_history.append(response)

                print(f"Response: {response}, '\n")
                print(f"Response Additional Kwargs: {response.additional_kwargs}, '\n")
                
                if response.content != '' : 
                    st.markdown(response.content)
                    st.session_state.messages.append({"role": "assistant", "content": response.content})

                if response.content == '':
                    api_requests_and_responses = []
                    backend_details = ''
                    
                    try:
                        params = {}
                        # Extracting information
                        tool_calls = response.additional_kwargs.get('tool_calls', [])
                        for call in tool_calls:
                            function_id = call.get('id')
                            function = call.get('function', {})
                            function_name = function.get('name')
                            function_arg = json.loads(function.get('arguments'))
                    
                        print(function_name)
                        print(function_arg)
                        #  EXTRACT THE PARAMETERS TO BE PASSED TO THE FUNCTIONS
                        for key,value in function_arg.items():
                            params[key] = value
                            print(function_name)
                            print(params)
                            print(params[key])
                        # PERFORMS THE FUNCTION CALL OUTSIDE THE LLM MODEL
                        if function_name == 'WhatsAppChatsInput':
                            with st.status('Analyzing your WhatsApp Chats', expanded=True) as status:
                                api_response = rag_class.retriever(params[key])
                                status.update(label='Analysis Complete', state='complete', expanded=False)
                            
            
                        # PARSE THE RESPONSE OF THE API CALLS BACK INTO THE MODEL
                        for k, v in api_response.items(): # check is reponse contains bytes object
                            if isinstance(v, BytesIO):
                                
                                # image = display_image(v)
                                # st.image(image=image,width=200)
                                response = process_chat(chain, 
                                                        ToolMessage(content=k, name=function_name, tool_call_id=function_id),
                                                        st.session_state.chat_history)
                                print('======================================')
                                print(response)
                                st.markdown(response.content)
                        
                        
                        # APPEND THE FUNCTION RESPONSE AND AI RESPONSE TO BOTH THE CHAT_HISTORY AND MESSAGE HISTORY FOR STREAMLIT  
                        st.session_state.chat_history.append(ToolMessage(
                            content=k, name=function_name, tool_call_id=function_id)) # append tool response
                        
                        st.session_state.chat_history.append(response)
                        st.session_state.messages.append({"role": "assistant", "content": response.content})
                        
                    
                    except (Exception,LangChainException) as e:
                        print(f"An error occurred: {e}")
        
    
    

except NameError:
    st.error('Unable to load the Stats. Please Upload a WhatsApp Chats .txt file.')
    
# ---footer---

st.markdown(
    """
    <style>
        div.stMarkdown footer {  /* Target only the footer within stMarkdown */
            display: flex; 
            justify-content: center;
            align-items: center;
            padding: 25px; /* More padding for a comfortable feel */
            # background: linear-gradient(to right, #25D366, #128C7E); /* WhatsApp-like gradient */
            color: #DCF8C6; /* Lighter green for the text, matching WhatsApp bubbles */
            font-size: 18px;
            border-radius: 15px; 
            margin-top: 40px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); 
        }

        div.stMarkdown footer p {
            margin: 0; 
        }

        div.stMarkdown footer a {
            color: inherit; /* Inherit color from parent (footer), which is #DCF8C6 */
            text-decoration: none;
            font-weight: bold;
            position: relative; 
            transition: all 0.3s ease; 
        }

        div.stMarkdown footer a::after {
            content: "";
            position: absolute;
            bottom: -4px; 
            left: 0;
            width: 100%;
            height: 2px;
            background-color: #fff; 
            transform: scaleX(0); 
            transform-origin: left; 
            transition: transform 0.3s ease; 
        }

        div.stMarkdown footer a:hover::after {
            transform: scaleX(1); 
        }

        div.stMarkdown footer a:hover {
            color: #f5f5f5; 
            letter-spacing: 1px; 
        }
    </style>

    <footer>
        <p>
            Made with ❤️ by&nbsp;
            <a href="https://github.com/Abdulraqib20" target="_blank">raqibcodes</a>
        </p>
    </footer>
    """,
    unsafe_allow_html=True,
)
