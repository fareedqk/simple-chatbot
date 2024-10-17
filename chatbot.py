import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the tokenizer and model for Flan-T5-base
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

# Streamlit app title
st.title("AI Chatbot")
st.write("Ask anything about Artificial Intelligence!")

# Input text box for user to ask questions
user_input = st.text_input("You:", "")

if user_input:
    # Tokenize the input and generate a response
    inputs = tokenizer(user_input, return_tensors="pt")
    outputs = model.generate(inputs.input_ids, max_length=200, num_beams=5, early_stopping=True)
    
    # Decode the generated response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Display the bot response
    st.write("AI Bot:", response)
