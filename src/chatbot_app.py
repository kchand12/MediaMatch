import streamlit as st
import model_framework

# user_input = "movie recommendation"
# response= model_framework.response(user_input)
# print(response)
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'context' not in st.session_state:
    st.session_state.context = {}  # Initialize an empty context
if 'input_submitted' not in st.session_state:
    st.session_state.input_submitted = False  # Track if the input was submitted

# Function to handle user input and get a response from the model, using context
def get_bot_response(user_input, context):
    response, updated_context = model_framework.response(user_input, context=context)
    return response, updated_context

# Streamlit UI setup
st.title("Movie Recommendation Chatbot")

# Chat input
user_input = st.text_input("Chat with your bot here", key="chat_input", 
                           value="" if st.session_state.input_submitted else None)

# Submit button for the chat
if st.button('Send'):
    if user_input:
        # Append user input to chat history
        st.session_state.chat_history.append({"role": "user", "text": user_input})

        # Getting a response from the bot, along with the updated context
        chat_response, updated_context = get_bot_response(user_input, st.session_state.context)

        # Update the context in session state
        st.session_state.context = updated_context

        # Append bot response to chat history
        st.session_state.chat_history.append({"role": "assistant", "text": chat_response})

        # Indicate that input was submitted to clear the chat input box
        st.session_state.input_submitted = True

        st.session_state.current_input = " "

# Reset input_submitted state after rendering to allow for new input
st.session_state.input_submitted = False

# Clear chat history button
if st.button('Clear Chat'):
    st.session_state.chat_history = []
    
# Display the chat history
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["text"])
