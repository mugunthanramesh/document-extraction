import streamlit as st
from langchain.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# Initialize the Local Model
llm = Ollama(model="mistral")

# Set up Memory for Conversation
memory = ConversationBufferMemory()

# Create a Conversation Chain
conversation = ConversationChain(
    llm=llm,
    memory=memory
)

# Streamlit UI
st.title("Chat with LangChain & Ollama")
st.write("Type your message below and chat with the AI.")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Scrollable chat history
chat_container = st.container()
with chat_container:
    for message in st.session_state["messages"]:
        st.write(message)

# Fixed position container for file uploader and input at the bottom
st.markdown(
    """
    <style>
        .fixed-input-container {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: white;
            padding: 10px;
            z-index: 9999;
            box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.1);
        }
        .stChatContainer {
            height: 70vh;
            overflow-y: auto;
            padding-bottom: 80px; /* Space for input box */
        }
    </style>
    """,
    unsafe_allow_html=True
)

with st.container():
    st.markdown('<div class="fixed-input-container">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a file", type=["txt", "pdf", "docx"])
    user_input = st.text_input("You:", key="user_input")
    st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    st.write("File uploaded successfully!")

if st.session_state.get("trigger", False):
    user_input = st.session_state["user_input"]
    st.session_state["messages"].append(f"You: {user_input}")
    st.rerun()

if user_input:
    st.write(f"You: {user_input}")
    response = conversation.run(user_input)
    st.session_state["messages"].append(f"You: {user_input}")
    st.session_state["messages"].append(f"Bot: {response}")
    st.write(f"Bot: {response}")

