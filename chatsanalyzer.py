# Import libraries and packages
import streamlit as st
import pandas as pd
import numpy as np
import base64
import plotly.express as px
import altair as alt

import re
from collections import Counter
import emoji
from io import BytesIO
import xlsxwriter
import collections
import datetime
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import nltk 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist

import warnings
warnings.filterwarnings(action='ignore')

from viz import show_viz

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

# --- How To Use ---
with st.container():
    with st.expander("How To Use"):
        st.markdown(
            """
            <div class="section get-started-section">
                <p>
                    This Application is a simple and easy-to-use WhatsApp Chats Analysis tool, thoughtfully designed and developed by Raqib (raqibcodes).
                    This application offers you a delightful and straightforward way to analyze your WhatsApp conversations. Dive into your chats, uncover valuable insights,
                    and gain a deeper understanding of your messaging history. Whether you're curious about your most active group members, most active times and other amazing stats,
                    this tool has got you covered. It's not just a utility; it's an exciting journey through your messages. Share this incredible experience with your friends and let the fun begin!😎
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

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
                        st.markdown(f'<div style="max-height: 300px; overflow-y: scroll;">{pd.DataFrame(unique_values, columns=[column]).to_html(index=False)}</div>', unsafe_allow_html=True)

            # Export the dataset to Excel format
            if st.button("Export Data to Excel File"):
                # Create a BytesIO buffer for writing the Excel file
                excel_buffer = BytesIO()
                with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
                    df.to_excel(writer, sheet_name="WhatsApp_Chat", index=False)

                # Set the filename and download button label
                excel_filename = "whatsapp_chats.xlsx"
                st.write(f"Exporting to {excel_filename}...")

                # Prepare the Excel data for download
                excel_data = excel_buffer.getvalue()
                st.download_button(label="Click here to download the Excel file", data=excel_data, file_name=excel_filename, key="excel_download")

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



###----------------------------------------VISUALIZATIONS--------------------------------------------------------------####
    #-----------------------------------------------STYLING TABS-------------------------------------------------------------#

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
    
    show_viz(df)
    

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
