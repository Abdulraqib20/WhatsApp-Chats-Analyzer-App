# Import libraries and packages
import streamlit as st
import pandas as pd
import numpy as np
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
from nltk.probability import FreqDist

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)


# Create a Streamlit app
st.set_page_config(
    page_title="Raqib's WhatsApp Chats Analyzer",
    page_icon="icons8-whatsapp-48.png",
    layout="wide",
)

# Center-align subheading and image using HTML <div> tags
st.markdown(
    """
    <div style="display: flex; flex-direction: column; align-items: center; text-align: center;">
        <h2>WhatsApp Chats Analyzer</h2>

    </div>
    """,
    unsafe_allow_html=True
)
st.image("rachit-tank-lZBs-lD9LPQ-unsplash.jpg")


# Add an introductory paragraph
st.markdown("""
This Application is a simple and easy-to-use WhatsApp Chats Analysis tool, thoughtfully designed and developed by Raqib, known as raqibcodes. This application offers you a delightful and straightforward way to analyze your WhatsApp conversations. Dive into your chats, uncover valuable insights, and gain a deeper understanding of your messaging history. Whether you're curious about your most active group members, most active times and other amazing stats, this tool has got you covered. It's not just a utility; it's an exciting journey through your messages. Share this incredible experience with your friends and let the fun begin!😎
""")

# Display a GIF image with a caption and custom dimensions
st.caption("Demo on how to export WhatsApp chats to Text File")
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

# Create a file upload widget
uploaded_file = st.file_uploader("Upload your WhatsApp chat .txt file", type=["txt"])


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
    st.subheader(f"Stats for {selected_member}")
    member_df = df[df['member'] == selected_member]
    
    # Calculate Messages Sent
    messages_sent = member_df.shape[0]

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
    st.write(f"Messages Sent: {messages_sent}")
    st.write(f"Words per Message: {words_per_message}")
    st.write(f"Media Messages Sent: {media_messages_sent}")
    st.write(f"Emojis Sent: {emojis_sent}")
    st.write(f"Links Sent: {links_sent}")

st.title('Visualizations')
# Most Active participants

participant_counts = df['member'].value_counts()
# Count the number of messages per member
message_counts = df['member'].value_counts().reset_index()
message_counts.columns = ['member', 'message count']

with st.expander("Most Active Participants", expanded=True):
    show_all_participants = st.checkbox("Show All Participants", value=True)
    
    if not show_all_participants:
        max_participants = st.slider("Max Participants to Show", min_value=1, max_value=len(message_counts), value=len(message_counts))
        message_counts = message_counts.head(max_participants)

# Create an Altair bar chart
chart = alt.Chart(message_counts).mark_bar().encode(
    x=alt.X('member:N', title='Participant', sort='-y'), 
    y=alt.Y('message count:Q', title='Number of Messages'),
    color=alt.Color('member:N', legend=None),
    tooltip=['member:N', 'message count:Q']
).properties(
    width=800,
    height=550,
    title='Most Active Participants by Message Count'
).configure_axisX(
    labelAngle=-45
)

st.altair_chart(chart, use_container_width=True)


# Emoji dist: Extract all emojis used in the chat and count their occurrences

# Calculate emoji frequencies
total_emojis_list = list([a for b in df['emojis'] for a in b])
emoji_dict = dict(collections.Counter(total_emojis_list))
emoji_dict = sorted(emoji_dict.items(), key=lambda x: x[1], reverse=True)
emoji_df = pd.DataFrame(emoji_dict, columns=['Emoji', 'Frequency'])
emoji_df = emoji_df.head(40)

# Create a list of just the emojis
emojis_only = emoji_df['Emoji']

# Define custom colors for the bar chart
colors = ['#ffd700', '#ff69b4', '#1e90ff', '#ff8c00', '#00ced1']

# Create a Streamlit expander for displaying the emoji distribution chart
with st.expander("Emoji Distribution", expanded=True):
    # Create a Plotly bar chart of emoji frequencies with custom colors
    fig = px.bar(
        emoji_df,
        x='Frequency',
        y=emojis_only,  
        orientation='h',  
        title='Overall Emoji Distribution',
        color=emojis_only, 
        color_discrete_sequence=colors,
    )
    
    # Customize the chart layout
    fig.update_layout(width=800, height=500, showlegend=True)
    
    # Display the chart using Plotly
    st.plotly_chart(fig)

# Most commonly Used Words
# Filter out messages that contain media files
non_media = df[~df['message'].str.contains('<Media omitted>')]

with st.expander('Most Used Words'):
    all_messages = ' '.join(non_media['message'].astype(str).tolist())
    all_words = all_messages.split()
    word_freq = collections.Counter(all_words)
    top_words = word_freq.most_common(20)
    words = [word[0] for word in top_words]
    counts = [count[1] for count in top_words]
    
    fig = px.bar(pd.DataFrame({'Word': words, 'Count': counts}), x='Word', y='Count')
    fig.update_xaxes(tickangle=0) 
    
    st.plotly_chart(fig)


# Word CLoud
with st.expander('Word Cloud of Messages'):
    all_messages = ' '.join(non_media['message'].astype(str).tolist())
    all_words = all_messages.split()
    word_freq = collections.Counter(all_words)

    # Create a WordCloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_messages)
    fig = px.imshow(wordcloud.to_array())
    fig.update_layout(title_text="Word Cloud")
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    st.plotly_chart(fig)
    
# Most Active Dates

with st.expander('Most Active Dates'):
    activity_by_date = df['date'].value_counts().reset_index()
    activity_by_date.columns = ['Date', 'Activity Count']
    
    fig = px.bar(activity_by_date, x='Date', y='Activity Count', title='Most Active Dates')
    fig.update_traces(marker_color='rgb(63, 72, 204)') 
    fig.update_xaxes(categoryorder='total descending')
    st.plotly_chart(fig)

# Most active times

with st.expander("Most Active Times", expanded=True):
    fig = px.line(
        df['time'].value_counts().head(20).reset_index(),
        x='index',  # Time values
        y='time',   # Message counts
        labels={'index': 'Time of Day', 'time': 'Number of Messages'},
        title='Most Active Times',
    )

    fig.update_xaxes(title_text='Time of Day')
    fig.update_yaxes(title_text='Number of Messages')
    fig.update_layout(width=850, height=550)
    st.plotly_chart(fig)
    
# Most active hour of the Day

with st.expander('Most Active Hours of the Day'):
    df['hour'] = df['time'].str.split(':', expand=True)[0]
    time_counts = df['hour'].value_counts().reset_index().rename(columns={'index': 'hour', 'hour': 'number of messages'})
    time_counts = time_counts.sort_values(by='hour')
    
    fig = px.bar(time_counts, x='hour', y='number of messages', color='hour',
                 title='Most Active Times (Hourly)')
    fig.update_layout(xaxis_title='Hour of the Day', yaxis_title='Number of Messages', showlegend=False)
    st.plotly_chart(fig)
    
# Most active times
# with st.expander('Most Active Times'):
#     time_counts = df['time'].value_counts().head(20).reset_index().rename(columns={'index': 'time', 'time': 'count'})
#     fig = px.bar(time_counts, x='count', y='time', orientation='h', color='time',
#                  title='Most Active Times of the Day')
#     fig.update_layout(xaxis_title='Number of Messages', yaxis_title='Time', showlegend=False)
    
#     st.plotly_chart(fig)

    
# Most active Days of the Week
with st.expander('Most Active Days of the Week'):
    df['weekday'] = df['date'].dt.day_name()
    day_counts = df['weekday'].value_counts().reset_index().rename(columns={'index': 'weekday', 'weekday': 'messages'})
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_counts['weekday'] = pd.Categorical(day_counts['weekday'], categories=days_order, ordered=True)
    day_counts = day_counts.sort_values('weekday')

    fig = px.bar(day_counts, x='messages', y='weekday', orientation='h', color='weekday',
                 title='Most Active Days of the Week')
    fig.update_layout(xaxis_title='Number of Messages', yaxis_title='Day of the Week', showlegend=False)
    st.plotly_chart(fig)

with st.expander('Most Active Days of the Week'):
    df['weekday'] = pd.to_datetime(df['date']).dt.day_name()
    day_counts = df['weekday'].value_counts().reset_index().rename(columns={'index': 'weekday', 'weekday': 'messages'})
    
    if 'weekday' in day_counts:
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_counts['weekday'] = pd.Categorical(day_counts['weekday'], categories=days_order, ordered=True)
        day_counts = day_counts.sort_values('weekday')

        # Create an Altair bar chart
        chart = alt.Chart(day_counts).mark_bar().encode(
            x='messages:Q',
            y=alt.Y('weekday:N', sort=days_order),
            color=alt.Color('weekday:N', legend=None)
        ).properties(
            title='Most Active Days of the Week'
        ).configure_axis(
            titleFontSize=14,
            labelFontSize=12
        )
        
        st.altair_chart(chart, use_container_width=True)
    else:
        st.warning("No data available for Most Active Days of the Week.")
    
# Messages Sent Per Month
with st.expander('Messages Sent Per Month'):
    df['month'] = pd.to_datetime(df['date']).dt.strftime('%B')
    messages_per_month = df['month'].value_counts().reset_index()
    messages_per_month.columns = ['Month', 'Messages Sent']
    months_order = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    messages_per_month['Month'] = pd.Categorical(messages_per_month['Month'], categories=months_order, ordered=True)
    messages_per_month = messages_per_month.sort_values(by='Month')
    
    fig = px.bar(messages_per_month, x='Month', y='Messages Sent', title='Messages Sent Per Month')
    fig.update_traces(marker_color='rgb(63, 72, 204)')
    fig.update_layout(xaxis_title='Month', yaxis_title='Messages Sent', font=dict(size=14))
    
    st.plotly_chart(fig, use_container_width=True)
    
# Visualize message count over time
with st.expander('Messages Over Time'):
    message_count_over_time = df.groupby(['date']).size().reset_index(name='messages')
    fig = px.line(message_count_over_time, x='date', y='messages', title='Messages Over Time')
    st.plotly_chart(fig)
    
# Visualize message length distribution
with st.expander('Message Length Distribution'):
    fig = px.histogram(df, x='message_length', title='Message Length Distribution')
    st.plotly_chart(fig)
    
# Member Activity Over Time
with st.expander('Member Activity Over Time'):
    member_activity_over_time = df.groupby(['date', 'member']).size().reset_index(name='messages')
    fig = px.line(member_activity_over_time, x='date', y='messages', color='member', title='Member Activity Over Time')
    st.plotly_chart(fig)

    
# footer

# line separator
st.markdown('<hr style="border: 2px solid #ddd;">', unsafe_allow_html=True)

# footer text
st.markdown(
    """
    <div style="text-align: center; padding: 10px;">
        Developed by <a href="https://github.com/Abdulraqib20" target="_blank">raqibcodes</a>
    </div>
    """,
    unsafe_allow_html=True
)
