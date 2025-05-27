import streamlit as st


from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

import os
from dotenv import load_dotenv

#get the key from .env file
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")


st.title("Chatbot with context retention")
st.write("By: Krishna, 2024A7PS0065H for ACM BPHC")


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize conversation chain
if "conversation" not in st.session_state:
    if not api_key:
        st.error("Please set your GROQ_API_KEY in the .env file")
        st.stop()
    
    
    # Specify LLM details
    llm = ChatGroq(
        model_name="llama3-8b-8192",
        temperature=0.7,
        groq_api_key=api_key
    )
    
    # Creating memory buffer using langchain
    memory = ConversationBufferMemory()
    
    # Create chain that connects LLM with memory
    st.session_state.conversation = ConversationChain(
        llm=llm, 
        memory=memory
    )

# Displaying chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Taking Chat input and Displaying input msg
if user_input := st.chat_input("Message"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Get AI response using conversation chain
    with st.chat_message("assistant"):
        ai_response = st.session_state.conversation.predict(input=user_input)
        st.markdown(ai_response)
        
        # Add AI response to display
        st.session_state.messages.append({"role": "assistant", "content": ai_response})