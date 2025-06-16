import os
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import torch

st.set_page_config(page_title="🧠 Local Mistral Chat", layout="centered")
# Authenticate with Hugging Face token (required for gated models like Mistral)
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    login(HF_TOKEN)
else:
    st.error("Missing HF_TOKEN. Please pass your Hugging Face access token as an environment variable.")
    st.stop()

# Load the model and tokenizer using Streamlit cache
@st.cache_resource
def load_model():
    model_id = "mistralai/Mistral-7B-Instruct-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        use_auth_token=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    return tokenizer, model

tokenizer, model = load_model()


st.title("💬 Mistral 7B Instruct Chatbot")
st.caption("Running locally with GPU + Hugging Face token auth")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_input = st.chat_input("Ask something...")

if user_input:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            # Format full conversation as prompt with special tokens
            prompt = "\n".join(
                [f"<|user|>{m['content']}" if m["role"] == "user" else f"<|assistant|>{m['content']}" for m in st.session_state.messages]
            ) + "\n<|assistant|>"

            input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)

            output = model.generate(
                **input_ids,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )

            decoded = tokenizer.decode(output[0], skip_special_tokens=True)
            response = decoded[len(prompt):].strip()

            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
