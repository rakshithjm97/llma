import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Hugging Face access token
token = "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"

# Load model and tokenizer from Hugging Face and use GPU if available
@st.cache_resource()
def load_model():
    model_name = "meta-llama/Llama-3.2-1B"  # Llama model from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
    
    # Use GPU if available for faster inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=token).to(device)
    return model, tokenizer, device

# Generate chat response using the model with optimized sampling parameters
def generate_response(model, tokenizer, prompt, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Speed up generation by reducing max_length and controlling top_k, top_p, etc.
    outputs = model.generate(
        **inputs, 
        max_length=100,  # Reduce length for faster output
        do_sample=True, 
        top_k=40,        # Decrease top_k for faster decoding
        top_p=0.9,       # Adjust to allow only higher-probability tokens
        temperature=0.8  # Slightly higher temperature to improve randomness
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Load the external CSS file
def load_css():
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Streamlit app UI
def main():
    st.title("Llama-3.2-1B Chatbot")

    # Load the CSS file
    load_css()

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Load the model, tokenizer, and device (CPU/GPU)
    model, tokenizer, device = load_model()

    # Create a form for user input
    with st.form("chat_form"):
        user_input = st.text_input("You:", "")
        submit_button = st.form_submit_button("Send")

    # Check if the submit button is clicked and there's user input
    if submit_button and user_input:
        # Add user message to the chat history
        st.session_state["messages"].append(f"**You**: {user_input}")
        st.write(f"**You**: {user_input}")
        
        # Generate and display model response
        with st.spinner("Llama-3.2-1B is typing..."):
            response = generate_response(model, tokenizer, user_input, device)
            st.session_state["messages"].append(f"**Llama-3.2-1B**: {response}")
            st.write(f"**Llama-3.2-1B**: {response}")
    
    # Display chat history from session state
    if st.session_state["messages"]:
        st.markdown("---")  # Separator line
        for message in st.session_state["messages"]:
            st.write(message)

if __name__ == "__main__":
    main()
