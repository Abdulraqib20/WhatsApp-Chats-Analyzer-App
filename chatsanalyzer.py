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

# # Sentiment Analysis
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch.nn.functional as F
# from torch.nn.functional import softmax
# import torch
# from sklearn.model_selection import train_test_split
# import requests
# from bs4 import BeautifulSoup
# import string

import warnings
warnings.filterwarnings(action='ignore')


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
        .main-header {
            background: linear-gradient(to right, #25D366, #128C7E); /* WhatsApp-like gradient */
            color: white; /* White text for the header */
            padding: 5px; /* Increased padding for more space */
            text-align: center;
            border-radius: 13px; /* Rounded corners for a softer look */
            box-shadow: 2px 2px 5px rgba(0,0,0,0.1); /* Subtle shadow for depth */
        }

        .main-header h1 {
            font-size: 2.2rem; /* Larger font size for the header */
        }

        /* Style for the "rocket" emoji */
        .main-header h1 span {
            animation: rocket-animation 2s linear infinite; /* Add animation */
        }

        @keyframes rocket-animation {
            0% { transform: translateY(0); }
            50% { transform: translateY(-10px); }
            100% { transform: translateY(0); }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Header ---


with open("icons8-whatsapp-48.png", "rb") as f:  
    favicon_data = f.read()
    b64_favicon = base64.b64encode(favicon_data).decode()

# --- Header ---
st.markdown(
    f"""
    <style>
        .whatsapp-icon {{
            height: 50px; 
            margin-right: 15px;
            vertical-align: middle;
        }}
    </style>

    <div class="main-header">
        <h1><img src="data:image/png;base64,{b64_favicon}" class="whatsapp-icon">WhatsApp Chats Analyzer</h1> 
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
        .intro-section, .get-started-section {
            # background-color: #fff; /* White background */
            border: 1px solid #ddd; /* Subtle border */
            padding: 25px; /* More padding for better readability */
            border-radius: 10px; /* Rounded corners for a softer look */
            box-shadow: 2px 2px 5px rgba(0,0,0,0.1); /* Subtle shadow for depth */
        }

        .intro-section h2, .get-started-section h3 {
            color: #fffff;
            margin-bottom: 15px;
        }

        .intro-section p, .get-started-section p {
            line-height: 1.6;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- How To Use ---
with st.container():
    with st.expander("How To Use"):  # Wrap the content in an expander
        st.markdown(
            """
            <div class="get-started-section">
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

# --- Styling ---
st.markdown(
    """
    <style>
        /* Existing styles... */

        .get-started-section {
            padding: 15px; /* Reduce padding a bit when inside the expander */
            border: none;   /* Remove the border to look cleaner within the expander */
            box-shadow: none;
        }

        .stExpanderHeader { /* Style the expander header */
            background-color: #007bff; /* Blue background */
            color: white;
            padding: 10px 15px; /* Adjust padding to your preference */
            font-weight: bold;
            border-radius: 5px; /* Rounded corners */
            cursor: pointer; /* Indicate it's clickable */
        }

        .stExpanderContent p {
            margin-top: 0; /* Remove extra margin at the top of the content */
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title(" ")
# Display a GIF image with a caption and custom dimensions
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
uploaded_file = st.file_uploader("Upload your WhatsApp chats .txt file", type=["txt"])

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

    st.title('Visualizations & Charts')

    # Initialize session state variable for Expanders
    
    if 'expanders_state' not in st.session_state:
        st.session_state.expanders_state = False
    toggle_button = st.button("Toggle Expanders")  # Button to toggle expanders
    
    if toggle_button:      # Update session state variable on button click
        st.session_state.expanders_state = not st.session_state.expanders_state

    # Most Active participants

    participant_counts = df['member'].value_counts()
    
    # Count the number of messages per member
    message_counts = df['member'].value_counts().reset_index()
    message_counts.columns = ['member', 'message count']
    
    with st.expander("Group Participants Overview", expanded=st.session_state.expanders_state):
        option = st.radio("Select Participants", ["Top", "Bottom"])
        num_participants = st.number_input(f"{option} Participants", min_value=1, max_value=len(message_counts), value=min(10, len(message_counts)))
        # Choose a label based on the user's selection
        label = st.text_input(f"Enter a title for the {option} Group participants", value=f"{option} {num_participants}")
    
        # Modify the logic to display the top/bottom N participants based on user input
        if option == "Top":
            message_counts = message_counts.nlargest(num_participants, 'message count')
        elif option == "Bottom":
            message_counts = message_counts.nsmallest(num_participants, 'message count')
    
        # Create an Altair bar chart
        chart = alt.Chart(message_counts).mark_bar().encode(
            x=alt.X('message count:Q', title='Number of Messages'),
            y=alt.Y('member:N', title='Participant', sort='-x'),
            color=alt.Color('member:N', legend=None),
            tooltip=['member:N', 'message count:Q']
        ).properties(
            width=500,
            height=550,
            title=f'{label} Participants by Number of Messages'
            ).configure_axis(
            grid=False
            )
        st.altair_chart(chart, use_container_width=True)
    
        # Participants Overview (Pie Chart)
            
        # Calculate the percentage of total messages
        total_messages = message_counts['message count'].sum()
        message_counts['Percentage'] = (message_counts['message count'] / total_messages) * 100
    
        # Create a donut chart using Plotly Express
        fig = px.pie(
            message_counts,
            names='member',
            values='message count',
            hole=0.4,
            title=f'{label} Participants by Number of Messages',
            labels={'member': 'Participant', 'message count': 'Number of Messages'},
            hover_data=['Percentage'],
            template='plotly',
            color_discrete_sequence=px.colors.qualitative.Set1,
        )
    
        # Display the donut chart using Plotly Express
        st.plotly_chart(fig)


    # Emoji dist: Extract all emojis used in the chat and count their occurrences
    
    # Calculate emoji frequencies
    total_emojis_list = [a for b in df['emojis'] for a in b]
    emoji_counter = Counter(total_emojis_list)
    emoji_df = pd.DataFrame(emoji_counter.items(), columns=['Emoji', 'Frequency']).sort_values(by='Frequency', ascending=False)

   # Create a Streamlit expander for displaying the emoji distribution
    with st.expander("Emoji Distribution", expanded=st.session_state.expanders_state):
        num_emojis = st.slider(f"Number of Emojis to Display", min_value=1, max_value=len(emoji_df), value=10)
        
        # Choose a label based on the user's selection
        option = "Top" 
        label = st.text_input(f"Enter a title for {option} emojis", value=f"{option} {num_emojis}")
    
        if st.checkbox("Show Bottom Emojis", value=False):
            option = "Bottom"
            label = st.text_input(f"Enter a title for {option} emojis", value=f"{option} {num_emojis}")
            emoji_df = emoji_df.nsmallest(num_emojis, 'Frequency')
        else:
            emoji_df = emoji_df.nlargest(num_emojis, 'Frequency')
    
        # Create a horizontal bar chart using Plotly Express
        fig = px.bar(
            emoji_df,
            x='Frequency',
            y='Emoji',
            orientation='h',
            title=f'{label} Emojis Distribution Bar Chart',
            labels={'Emoji': 'Emoji', 'Frequency': 'Frequency'},
            color='Emoji',
            color_discrete_map=dict(zip(emoji_df['Emoji'], px.colors.qualitative.Set1))
        )
    
        # Display the bar chart
        st.plotly_chart(fig, use_container_width=True)

    # Most commonly Used Words
    
    # Filter out messages that contain media files
    non_media = df[~df['message'].str.contains('<Media omitted>')]

    with st.expander('Most Used Words', expanded=st.session_state.expanders_state):
        # Filter out messages that contain media files
        non_media = df[~df['message'].str.contains('<Media omitted>')]
    
        option = st.radio("Select Words", ["Top", "Bottom"])
        num_words = st.number_input(f"{option} Words", min_value=1, value=10)
    
        # Calculate word frequencies
        all_messages = ' '.join(non_media['message'].astype(str).tolist())
        all_words = all_messages.split()
        word_counter = collections.Counter(all_words)
    
        if option == "Top":
            top_words = word_counter.most_common(num_words)
        elif option == "Bottom":
            top_words = word_counter.most_common()[-num_words:]
    
        # Sort words in descending order of magnitude
        top_words = sorted(top_words, key=lambda x: x[1], reverse=False)
    
        words_df = pd.DataFrame(top_words, columns=['Word', 'Frequency'])
    
        # Create a bar chart using Plotly Express
        fig = px.bar(
            words_df,
            y='Word',
            x='Frequency',
            title=f'{option} {num_words} Used Words Chart',
            labels={'Word': 'Word', 'Frequency': 'Frequency'},
        )
    
        # Display the chart using Plotly
        st.plotly_chart(fig)


    # Word CLoud
    
    with st.expander('Word Cloud of Messages', expanded=st.session_state.expanders_state):
        all_messages = ' '.join(non_media['message'].astype(str).tolist())
        all_words = all_messages.split()
        word_freq = collections.Counter(all_words)
        
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_messages)
        st.image(wordcloud.to_image())

    # Most Active Dates

    with st.expander('Most Active Dates', expanded=st.session_state.expanders_state):
        activity_by_date = df['date'].value_counts().reset_index()
        activity_by_date.columns = ['Date', 'Activity Count']

        fig = px.bar(activity_by_date, x='Date', y='Activity Count', title='Most Active Dates')
        fig.update_traces(marker_color='rgb(63, 72, 204)') 
        fig.update_xaxes(categoryorder='total descending')
        st.plotly_chart(fig)

    # Most active times

    with st.expander("Most Active Times", expanded=st.session_state.expanders_state):
        counts = df.groupby('time').size().nlargest(20).reset_index(name='count')
    
        fig = px.line(
            counts,
            x='time',
            y='count',
            labels={'time': 'Time of Day', 'count': 'Number of Messages'},
            title='Most Active Times',
        )
    
        fig.update_xaxes(title_text='Time of Day')
        fig.update_yaxes(title_text='Number of Messages')
        fig.update_layout(width=850, height=550)
        st.plotly_chart(fig)

    # Most active hour of the Day

    with st.expander('Most Active Hours of the Day', expanded=st.session_state.expanders_state):
        df['hour'] = df['time'].str.split(':', expand=True)[0]
        time_counts = df.groupby('hour').size().reset_index(name='number of messages').sort_values(by='hour')
        
        fig = px.bar(
            time_counts,
            x='hour',
            y='number of messages',
            color='hour',
            title='Most Active Times (Hourly)'
        )
        
        fig.update_layout(xaxis_title='Hour of the Day', yaxis_title='Number of Messages', showlegend=False)
        st.plotly_chart(fig)

    # Most active Days of the Week
    with st.expander('Most Active Days of the Week', expanded=st.session_state.expanders_state):
        df['weekday'] = df['date'].dt.day_name()
        day_counts = df.groupby('weekday').size().reset_index(name='messages')
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        # day_counts['weekday'] = pd.Categorical(day_counts['weekday'], categories=days_order, ordered=True)
        day_counts = day_counts.sort_values('weekday')
    
        fig = px.bar(
            day_counts,
            x='messages',
            y='weekday',
            orientation='h',
            color='weekday',
            title='Most Active Days of the Week'
        )
        
        fig.update_layout(xaxis_title='Number of Messages', yaxis_title='Day of the Week', showlegend=False)
        st.plotly_chart(fig)

    # Messages Sent Per Month
    with st.expander('Messages Sent Per Month', expanded=st.session_state.expanders_state):
        df['month'] = pd.to_datetime(df['date']).dt.strftime('%B')
        messages_per_month = df['month'].value_counts().reset_index()
        messages_per_month.columns = ['Month', 'Messages Sent']
        months_order = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
        messages_per_month['Month'] = pd.Categorical(messages_per_month['Month'], categories=months_order, ordered=True)
        messages_per_month = messages_per_month.sort_values(by='Month')

        fig = px.bar(messages_per_month, x='Month', y='Messages Sent', title='Messages Sent Per Month')
        fig.update_traces(marker_color='rgb(63, 72, 204)')
        fig.update_layout(
            xaxis_title='Month',
            yaxis_title='Messages Sent',
            font=dict(size=14),
            width=500,
            height=550
        )
        st.plotly_chart(fig, use_container_width=True)

    # Visualize message count over time
    with st.expander('Messages Over Time', expanded=st.session_state.expanders_state):
        message_count_over_time = df.groupby(['date']).size().reset_index(name='messages')
        fig = px.line(message_count_over_time, x='date', y='messages', title='Messages Over Time')
        st.plotly_chart(fig)

    # Visualize message length distribution
    with st.expander('Message Length Distribution', expanded=st.session_state.expanders_state):
        fig = px.histogram(df, x='message_length', title='Message Length Distribution')
        st.plotly_chart(fig)

    # Member Activity Over Time
    with st.expander('Member Activity Over Time', expanded=st.session_state.expanders_state):
        member_activity_over_time = df.groupby(['date', 'member']).size().reset_index(name='messages')
        fig = px.line(member_activity_over_time, x='date', y='messages', color='member', title='Member Activity Over Time')
        st.plotly_chart(fig)

# # Sentiment Analysis

#     # Function to preprocess a single text
#     def preprocess_text(text):
#         # Define the denoise_text function
#         def denoise_text(text):
#             text = strip_html(text)
#             return text
        
#         # Define the strip_html function
#         def strip_html(text):
#             soup = BeautifulSoup(text, "html.parser")
#             return soup.get_text()
        
#         # Apply denoising functions
#         text = denoise_text(text)
        
#         # Convert to lowercase
#         text = text.lower()
        
#         # Remove URLs, hashtags, mentions, and special characters
#         text = re.sub(r"http\S+|www\S+|@\w+|#\w+", "", text)
#         text = re.sub(r"[^\w\s]", "", text)
        
#         # Remove numbers/digits
#         text = re.sub(r'\b[0-9]+\b\s*', '', text)
        
#         # Remove punctuation
#         text = ''.join([char for char in text if char not in string.punctuation])
        
#         # Tokenize the text
#         tokens = word_tokenize(text)
        
#         # Remove stop words
#         stop_words = set(stopwords.words('english'))
#         tokens = [token for token in tokens if token not in stop_words]
        
#         # Lemmatize the words
#         lemmatizer = WordNetLemmatizer()
#         tokens = [lemmatizer.lemmatize(token) for token in tokens]
        
#         # Join tokens back into a single string
#         return ' '.join(tokens)

#     # calculate sentiment scoring
#     def sentiment_score(text, model, tokenizer, label_mapping={1: 'Negative', 2: 'Neutral', 3: 'Positive'}):
#         try:
#             # Tokenize the input text
#             tokens = tokenizer.encode(text, return_tensors='pt')
    
#             # Get model predictions
#             with torch.no_grad():
#                 result = model(tokens)
    
#             # Obtain predicted class index
#             predicted_index = torch.argmax(result.logits).item()
    
#             # Map scores to labels
#             if label_mapping is not None:
#                 predicted_label = label_mapping.get(predicted_index + 1, f'Class {predicted_index + 1}')
    
#             # Calculate confidence percentage
#             probabilities = softmax(result.logits, dim=1)
#             confidence_percentage = str(probabilities[0, predicted_index].item() * 100) + '%'
    
#             # Return results
#             return {
#                 'predicted_label': predicted_label,
#                 'predicted_index': predicted_index + 1,
#                 'confidence_percentage': confidence_percentage
#             }
    
#         except Exception as e:
#             return {
#                 'error': str(e)
#             }

#     # model name
#     model_name = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
#     # load directory of saved model
#     save_directory = r"C:\Users\user\Desktop\MACHINE LEARNING\Sentiment Analysis\New folder"
#     # load model from the local directory
#     tokenizer = AutoTokenizer.from_pretrained(save_directory)
#     model = AutoModelForSequenceClassification.from_pretrained(save_directory)

    # st.title('Sentiment Analysis')
    # df['message'] = df['message'].astype('str')
    # # Apply sentiment analysis to the entire 'df['message']' column
    # df['processed_message'] = df['message'].apply(preprocess_text)
    # df['sentiment_results'] = df['processed_message'].apply(lambda x: sentiment_score(x, model, tokenizer))
        
    # # Display sentiment analysis results in a DataFrame
    # st.title('Sentiment Analysis Results')
    # st.write(df[['message', 'sentiment_results']])
        
    # # Breakdown of sentiments
    # sentiment_counts = df['sentiment_results'].apply(lambda x: x.get('predicted_label')).value_counts()
    # st.subheader('Sentiment Breakdown:')
    # st.write(sentiment_counts)

except NameError:
    st.error('Unable to load the Stats. Please Upload a WhatsApp Chats .txt file.')
    
# ---footer---

# footer text
st.title(" ")

st.markdown(
    """
    <style>
        footer {
            display: flex; 
            justify-content: center;
            align-items: center;
            padding: 25px; /* More padding for a comfortable feel */
            background: linear-gradient(to right, #25D366, #128C7E); /* WhatsApp-like gradient */
            color: white;
            font-size: 18px;
            border-radius: 15px; /* Softer rounded corners */
            margin-top: 40px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); /* Subtle shadow for depth */
        }

        footer p {
            margin: 0; /* Remove default margin for better control */
        }

        footer a {
            color: #fff;
            text-decoration: none;
            font-weight: bold;
            position: relative; /* For positioning the pseudo-element */
            transition: all 0.3s ease; /* Smoother transitions for all properties */
        }

        footer a::after {
            content: "";
            position: absolute;
            bottom: -4px; /* Adjust position of underline */
            left: 0;
            width: 100%;
            height: 2px;
            background-color: #fff; /* White underline */
            transform: scaleX(0); /* Initially hidden */
            transform-origin: left; /* Animate from the left */
            transition: transform 0.3s ease; /* Smooth transition */
        }

        footer a:hover::after {
            transform: scaleX(1); /* Show underline on hover */
        }

        footer a:hover {
            color: #f5f5f5; /* Slightly lighter color on hover */
            letter-spacing: 1px; /* Subtle letter spacing increase on hover */
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
