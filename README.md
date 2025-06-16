# 🧠 Mistral 7B Local Chatbot

A fully self-hosted chatbot UI running Mistral 7B with GPU acceleration and Streamlit.
It builds as docker, sends to streamlit, downloads file from huggingface, and you're up and going.
Need docker desktop and a streamlit account. 

## 🛠 Features

- Locally served Mistral-7B-Instruct-v0.1 (Hugging Face gated repo)
- Streamlit frontend UI
- Hugging Face authentication via `.env`
- GPU support via Docker + CUDA
- Quick deployment with Docker

## 🚀 Quickstart

```bash
# Clone and enter
git clone https://github.com/yourusername/simple-chat-mistral.git
cd simple-chat-mistral

# Add your Hugging Face token
cp .env.example .env
# edit .env and paste in your HF_TOKEN

# Build and run
docker build -t mistral-gpu-chat .
docker run --gpus all --env-file .env -p 8501:8501 mistral-gpu-chat
