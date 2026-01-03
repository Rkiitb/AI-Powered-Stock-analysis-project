from graph import workflow 
import streamlit as st
import uuid


CONFIG = {'configurable': {'thread_id': 'thread-1'}}
# session state for message history
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []


for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

query = st.chat_input("Ask about a stock or market:")

if query:
    thread_id = str(uuid.uuid4())

    CONFIG = {
        "configurable": {
            "thread_id": thread_id
        }
    }
    st.session_state['message_history'].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.text(query)


    response=workflow.invoke({'user_query':query},config=CONFIG)
    st.session_state['message_history'].append({"role": "assistant", "content": response['final_result']})
    with st.chat_message("assistant"):
        st.markdown(response['final_result'])
