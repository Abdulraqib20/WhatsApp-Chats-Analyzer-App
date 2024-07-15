# Import libraries and packages
import streamlit as st
import pandas as pd
import plotly.express as px
import altair as alt
from collections import Counter
import collections
from wordcloud import WordCloud
import warnings;warnings.filterwarnings(action='ignore')

# def show_viz(df):
#     st.title('Visualizations & Charts')

#     # Initialize session state variable for Expanders
    
#     if 'expanders_state' not in st.session_state:
#         st.session_state.expanders_state = False
#     toggle_button = st.button("Toggle Expanders")  # Button to toggle expanders
    
#     if toggle_button:      # Update session state variable on button click
#         st.session_state.expanders_state = not st.session_state.expanders_state

#     # Most Active participants

#     participant_counts = df['member'].value_counts()
    
#     # Count the number of messages per member
#     message_counts = df['member'].value_counts().reset_index()
#     message_counts.columns = ['member', 'message count']
    
#     with st.expander("Group Participants Overview", expanded=st.session_state.expanders_state):
#         option = st.radio("Select Participants", ["Top", "Bottom"])
#         num_participants = st.number_input(f"{option} Participants", min_value=1, max_value=len(message_counts), value=min(10, len(message_counts)))
#         # Choose a label based on the user's selection
#         label = st.text_input(f"Enter a title for the {option} Group participants", value=f"{option} {num_participants}")
    
#         # Modify the logic to display the top/bottom N participants based on user input
#         if option == "Top":
#             message_counts = message_counts.nlargest(num_participants, 'message count')
#         elif option == "Bottom":
#             message_counts = message_counts.nsmallest(num_participants, 'message count')
    
#         # Create an Altair bar chart
#         chart = alt.Chart(message_counts).mark_bar().encode(
#             x=alt.X('message count:Q', title='Number of Messages'),
#             y=alt.Y('member:N', title='Participant', sort='-x'),
#             color=alt.Color('member:N', legend=None),
#             tooltip=['member:N', 'message count:Q']
#         ).properties(
#             width=500,
#             height=550,
#             title=f'{label} Participants by Number of Messages'
#             ).configure_axis(
#             grid=False
#             )
#         st.altair_chart(chart, use_container_width=True)
    
#         # Participants Overview (Pie Chart)
            
#         # Calculate the percentage of total messages
#         total_messages = message_counts['message count'].sum()
#         message_counts['Percentage'] = (message_counts['message count'] / total_messages) * 100
    
#         # Create a donut chart using Plotly Express
#         fig = px.pie(
#             message_counts,
#             names='member',
#             values='message count',
#             hole=0.4,
#             title=f'{label} Participants by Number of Messages',
#             labels={'member': 'Participant', 'message count': 'Number of Messages'},
#             hover_data=['Percentage'],
#             template='plotly',
#             color_discrete_sequence=px.colors.qualitative.Set1,
#         )
    
#         # Display the donut chart using Plotly Express
#         st.plotly_chart(fig)


#     # Emoji dist: Extract all emojis used in the chat and count their occurrences
    
#     # Calculate emoji frequencies
#     total_emojis_list = [a for b in df['emojis'] for a in b]
#     emoji_counter = Counter(total_emojis_list)
#     emoji_df = pd.DataFrame(emoji_counter.items(), columns=['Emoji', 'Frequency']).sort_values(by='Frequency', ascending=False)

#    # Create a Streamlit expander for displaying the emoji distribution
#     with st.expander("Emoji Distribution", expanded=st.session_state.expanders_state):
#         num_emojis = st.slider(f"Number of Emojis to Display", min_value=1, max_value=len(emoji_df), value=10)
        
#         # Choose a label based on the user's selection
#         option = "Top" 
#         label = st.text_input(f"Enter a title for {option} emojis", value=f"{option} {num_emojis}")
    
#         if st.checkbox("Show Bottom Emojis", value=False):
#             option = "Bottom"
#             label = st.text_input(f"Enter a title for {option} emojis", value=f"{option} {num_emojis}")
#             emoji_df = emoji_df.nsmallest(num_emojis, 'Frequency')
#         else:
#             emoji_df = emoji_df.nlargest(num_emojis, 'Frequency')
    
#         # Create a horizontal bar chart using Plotly Express
#         fig = px.bar(
#             emoji_df,
#             x='Frequency',
#             y='Emoji',
#             orientation='h',
#             title=f'{label} Emojis Distribution Bar Chart',
#             labels={'Emoji': 'Emoji', 'Frequency': 'Frequency'},
#             color='Emoji',
#             color_discrete_map=dict(zip(emoji_df['Emoji'], px.colors.qualitative.Set1))
#         )
    
#         # Display the bar chart
#         st.plotly_chart(fig, use_container_width=True)

#     # Most commonly Used Words
    
#     # Filter out messages that contain media files
#     non_media = df[~df['message'].str.contains('<Media omitted>')]

#     with st.expander('Most Used Words', expanded=st.session_state.expanders_state):
#         # Filter out messages that contain media files
#         non_media = df[~df['message'].str.contains('<Media omitted>')]
    
#         option = st.radio("Select Words", ["Top", "Bottom"])
#         num_words = st.number_input(f"{option} Words", min_value=1, value=10)
    
#         # Calculate word frequencies
#         all_messages = ' '.join(non_media['message'].astype(str).tolist())
#         all_words = all_messages.split()
#         word_counter = collections.Counter(all_words)
    
#         if option == "Top":
#             top_words = word_counter.most_common(num_words)
#         elif option == "Bottom":
#             top_words = word_counter.most_common()[-num_words:]
    
#         # Sort words in descending order of magnitude
#         top_words = sorted(top_words, key=lambda x: x[1], reverse=False)
    
#         words_df = pd.DataFrame(top_words, columns=['Word', 'Frequency'])
    
#         # Create a bar chart using Plotly Express
#         fig = px.bar(
#             words_df,
#             y='Word',
#             x='Frequency',
#             title=f'{option} {num_words} Used Words Chart',
#             labels={'Word': 'Word', 'Frequency': 'Frequency'},
#         )
    
#         # Display the chart using Plotly
#         st.plotly_chart(fig)


#     # Word CLoud
    
#     with st.expander('Word Cloud of Messages', expanded=st.session_state.expanders_state):
#         all_messages = ' '.join(non_media['message'].astype(str).tolist())
#         all_words = all_messages.split()
#         word_freq = collections.Counter(all_words)
        
#         wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_messages)
#         st.image(wordcloud.to_image())

#     # Most Active Dates

#     with st.expander('Most Active Dates', expanded=st.session_state.expanders_state):
#         activity_by_date = df['date'].value_counts().reset_index()
#         activity_by_date.columns = ['Date', 'Activity Count']

#         fig = px.bar(activity_by_date, x='Date', y='Activity Count', title='Most Active Dates')
#         fig.update_traces(marker_color='rgb(63, 72, 204)') 
#         fig.update_xaxes(categoryorder='total descending')
#         st.plotly_chart(fig)

#     # Most active times

#     with st.expander("Most Active Times", expanded=st.session_state.expanders_state):
#         counts = df.groupby('time').size().nlargest(20).reset_index(name='count')
    
#         fig = px.line(
#             counts,
#             x='time',
#             y='count',
#             labels={'time': 'Time of Day', 'count': 'Number of Messages'},
#             title='Most Active Times',
#         )
    
#         fig.update_xaxes(title_text='Time of Day')
#         fig.update_yaxes(title_text='Number of Messages')
#         fig.update_layout(width=850, height=550)
#         st.plotly_chart(fig)

#     # Most active hour of the Day

#     with st.expander('Most Active Hours of the Day', expanded=st.session_state.expanders_state):
#         df['hour'] = df['time'].str.split(':', expand=True)[0]
#         time_counts = df.groupby('hour').size().reset_index(name='number of messages').sort_values(by='hour')
        
#         fig = px.bar(
#             time_counts,
#             x='hour',
#             y='number of messages',
#             color='hour',
#             title='Most Active Times (Hourly)'
#         )
        
#         fig.update_layout(xaxis_title='Hour of the Day', yaxis_title='Number of Messages', showlegend=False)
#         st.plotly_chart(fig)

#     # Most active Days of the Week
#     with st.expander('Most Active Days of the Week', expanded=st.session_state.expanders_state):
#         df['weekday'] = df['date'].dt.day_name()
#         day_counts = df.groupby('weekday').size().reset_index(name='messages')
#         days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
#         # day_counts['weekday'] = pd.Categorical(day_counts['weekday'], categories=days_order, ordered=True)
#         day_counts = day_counts.sort_values('weekday')
    
#         fig = px.bar(
#             day_counts,
#             x='messages',
#             y='weekday',
#             orientation='h',
#             color='weekday',
#             title='Most Active Days of the Week'
#         )
        
#         fig.update_layout(xaxis_title='Number of Messages', yaxis_title='Day of the Week', showlegend=False)
#         st.plotly_chart(fig)

#     # Messages Sent Per Month
#     with st.expander('Messages Sent Per Month', expanded=st.session_state.expanders_state):
#         df['month'] = pd.to_datetime(df['date']).dt.strftime('%B')
#         messages_per_month = df['month'].value_counts().reset_index()
#         messages_per_month.columns = ['Month', 'Messages Sent']
#         months_order = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
#         messages_per_month['Month'] = pd.Categorical(messages_per_month['Month'], categories=months_order, ordered=True)
#         messages_per_month = messages_per_month.sort_values(by='Month')

#         fig = px.bar(messages_per_month, x='Month', y='Messages Sent', title='Messages Sent Per Month')
#         fig.update_traces(marker_color='rgb(63, 72, 204)')
#         fig.update_layout(
#             xaxis_title='Month',
#             yaxis_title='Messages Sent',
#             font=dict(size=14),
#             width=500,
#             height=550
#         )
#         st.plotly_chart(fig, use_container_width=True)

#     # Visualize message count over time
#     with st.expander('Messages Over Time', expanded=st.session_state.expanders_state):
#         message_count_over_time = df.groupby(['date']).size().reset_index(name='messages')
#         fig = px.line(message_count_over_time, x='date', y='messages', title='Messages Over Time')
#         st.plotly_chart(fig)

#     # Visualize message length distribution
#     with st.expander('Message Length Distribution', expanded=st.session_state.expanders_state):
#         fig = px.histogram(df, x='message_length', title='Message Length Distribution')
#         st.plotly_chart(fig)

#     # Member Activity Over Time
#     with st.expander('Member Activity Over Time', expanded=st.session_state.expanders_state):
#         member_activity_over_time = df.groupby(['date', 'member']).size().reset_index(name='messages')
#         fig = px.line(member_activity_over_time, x='date', y='messages', color='member', title='Member Activity Over Time')
#         st.plotly_chart(fig)
        

#-----------------------------------------------STYLING TABS-------------------------------------------------------------#
# Inject custom CSS for styling the tabs
# st.markdown("""
#     <style>
#     /* Tab container */
#     div[role="tablist"] {
#         display: flex;
#         justify-content: space-around;
#         margin-bottom: 10px;
#         border-bottom: 2px solid #f0f0f0;
#     }

#     /* Individual tab */
#     div[role="tab"] {
#         padding: 10px 20px;
#         font-size: 16px;
#         cursor: pointer;
#     }

#     /* Active tab */
#     div[aria-selected="true"] {
#         color: #ffffff;
#         background-color: #4CAF50;  /* Change to your preferred active tab color */
#         border-radius: 10px 10px 0 0;
#         border: 1px solid #ddd;
#         border-bottom: none;
#     }

#     /* Inactive tab */
#     div[role="tab"]:not([aria-selected="true"]) {
#         color: #000000;
#         background-color: #e0e0e0;  /* Change to your preferred inactive tab color */
#         border: 1px solid #ddd;
#         border-radius: 10px 10px 0 0;
#     }

#     /* Tab content container */
#     div.stTabs {
#         margin-top: 0px;
#         border: 1px solid #ddd;
#         padding: 20px;
#         border-radius: 0 10px 10px 10px;
#         background-color: #ffffff;
#     }
#     </style>
#     """, unsafe_allow_html=True)



def show_viz(df):
    st.header('Visualizations & Charts')

    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12 = st.tabs([
        "Group Participants Overview",
        "Emoji Distribution",
        "Most Commonly Used Words",
        "Word Cloud of Messages",
        "Most Active Dates",
        "Most Active Times of the Day",
        "Most Active Hours of the Day",
        "Most Active Days of the Week",
        "Messages Sent Per Month",
        "Messages Sent Over Time",
        "Message Length Distributions",
        "Member Activity Over Time",
    ])

#------------------------------------------------------------------------------------------------------------------------#
    with tab1:
        st.write("Content for Participants tab")
        # Most Active participants
        # Count the number of messages per member
        message_counts = df['member'].value_counts().reset_index()
        message_counts.columns = ['member', 'message count']

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
        
        
#------------------------------------------------------------------------------------------------------------------------#
    with tab2:
        # Emoji dist: Extract all emojis used in the chat and count their occurrences
    
        # Calculate emoji frequencies
        total_emojis_list = [a for b in df['emojis'] for a in b]
        emoji_counter = Counter(total_emojis_list)
        emoji_df = pd.DataFrame(emoji_counter.items(), columns=['Emoji', 'Frequency']).sort_values(by='Frequency', ascending=False)
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


#------------------------------------------------------------------------------------------------------------------------#
    with tab3:
        # Most commonly Used Words
        
        # Filter out messages that contain media files
        non_media = df[~df['message'].str.contains('<Media omitted>')]
        
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
        
#------------------------------------------------------------------------------------------------------------------------#
    with tab4:
        all_messages = ' '.join(non_media['message'].astype(str).tolist())
        all_words = all_messages.split()
        word_freq = collections.Counter(all_words)
        
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_messages)
        st.image(wordcloud.to_image())
        
        
#------------------------------------------------------------------------------------------------------------------------#
    with tab5:
        activity_by_date = df['date'].value_counts().reset_index()
        activity_by_date.columns = ['Date', 'Activity Count']

        fig = px.bar(activity_by_date, x='Date', y='Activity Count', title='Most Active Dates')
        fig.update_traces(marker_color='rgb(63, 72, 204)') 
        fig.update_xaxes(categoryorder='total descending')
        st.plotly_chart(fig)
    
#------------------------------------------------------------------------------------------------------------------------#
    with tab6:
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

#------------------------------------------------------------------------------------------------------------------------#
    with tab7:
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

#------------------------------------------------------------------------------------------------------------------------#
    with tab8:
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
    
    
#------------------------------------------------------------------------------------------------------------------------#
    with tab9:
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


#------------------------------------------------------------------------------------------------------------------------#
    with tab10:
        # Messages Over Time
        message_count_over_time = df.groupby(['date']).size().reset_index(name='messages')
        fig = px.line(message_count_over_time, x='date', y='messages', title='Messages Over Time')
        st.plotly_chart(fig)

#------------------------------------------------------------------------------------------------------------------------#
    with tab11:
        # Message Length Distribution
        fig = px.histogram(df, x='message_length', title='Message Length Distribution')
        st.plotly_chart(fig)

#------------------------------------------------------------------------------------------------------------------------#
    with tab12:
        # Member Activity Over Time
        member_activity_over_time = df.groupby(['date', 'member']).size().reset_index(name='messages')
        fig = px.line(member_activity_over_time, x='date', y='messages', color='member', title='Member Activity Over Time')
        st.plotly_chart(fig)
