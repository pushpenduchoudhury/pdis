import os
import torch
from PIL import Image
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from torchvision import transforms
import config.conf as conf
import torchvision.models.resnet
torch.serialization.add_safe_globals([torchvision.models.resnet.ResNet])
load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

def clear_messages():
    st.session_state.chat_messages = []

st.set_page_config(
    page_title = "PDIS",
    page_icon = "🌱",
    layout = "wide",
    initial_sidebar_state = "collapsed"
)

# Set Title
title_col = st.columns([0.8, 12, 0.5])
with title_col[0]:
    st.image(image = Path(conf.ASSETS_DIR, "logo.png"), use_container_width = True)
with title_col[1]:
    st.header("Plant Disease Identification System", divider = "rainbow", anchor = False)
with title_col[2]:
    st.write("")
    st.button(":material/mop:", on_click = lambda: clear_messages(), type = "secondary", use_container_width = True)


def get_ollama_models() -> list:
    import ollama
    ollama_client = ollama.Client()
    llm = ollama_client.list()
    model_list = [i['model'] for i in llm['models']]
    return model_list

available_models = {
    "groq" : ["llama3-8b-8192", "llama-3.1-8b-instant", "llama-3.3-70b-versatile", "llama3-70b-8192", "deepseek-r1-distill-llama-70b", "gemma2-9b-it", "mistral-saba-24b", "qwen/qwen3-32b", "qwen-qwq-32b"],
    "google_genai" : ["gemini-1.5-flash"],
    "ollama" : [],
}

# Select Model
input_col, output_col = st.columns([0.3, 0.7])

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []


input_col.markdown("#### :grey[Upload an image of a plant]")
uploaded_image = input_col.file_uploader("Upload an image of a Plant", accept_multiple_files = False, type = ["jpg", "jpeg"], label_visibility = "collapsed")
check_button = input_col.button("Detect Disease")

with input_col.expander("Model Selection"):
    st.markdown("#####  :grey[Model]", unsafe_allow_html = True)
    model_provider = st.radio("Model Provider", options = available_models.keys(), horizontal = True, label_visibility = "collapsed")

    if model_provider == "ollama":
        try:
            ollama_models = get_ollama_models()
            available_models["ollama"] = ollama_models
        except Exception as e:
            st.warning(f"ERROR: {e}", icon = "⚠️")
            st.stop()

        if len(ollama_models) == 0:
            st.info(f"""No models found...! Please download a model from Ollama library to proceed.

    Command:

    ollama pull <model name>
                    
    You can visit the website: https://ollama.com/library to get models names.
    """, icon = "⚠")
            st.stop()

    model:str = st.selectbox("LLM Model", options = available_models[model_provider], width = "stretch", label_visibility = "collapsed")


def get_llm(model, model_provider):
    llm = init_chat_model(model = model, model_provider = model_provider)
    return llm


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),        # Resize to model input size
        transforms.ToTensor(),                 # Convert to tensor
        transforms.Normalize(                  # Normalize with ImageNet stats
            mean = [0.485, 0.456, 0.406], 
            std = [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

def detect_disease(image):
    # input_tensor = preprocess_image(image)
    # classification_model = torch.load(conf.DISEASE_DETECTION_MODEL, weights_only = False)
    # classification_model.eval()

    # with torch.no_grad():
    #     outputs = classification_model(input_tensor)
    #     _, predicted = torch.max(outputs, 1)

    # predicted_class = conf.DISEASE_CLASSES[predicted.item()]
    predicted_class = "Tomato__blight"
    return predicted_class

# if uploaded_image:


if check_button:
    message_container = output_col.container(height = 450, border = True)
    with message_container:
        with st.chat_message("assistant"):
            st.image(uploaded_image)

    predicted_class = detect_disease(image = uploaded_image)

    if "healthy" in predicted_class:
        with message_container:
            with st.chat_message("assistant"):
                st.markdown("The plant is healthy")
    else:
        with message_container:
            with st.chat_message("assistant"):
                st.markdown(f'Disease Detected: {predicted_class.split("___")[1] if "___" in predicted_class else predicted_class.split("__")[1] if "__" in predicted_class else predicted_class.split("_")[1]}')
        
        system_prompt = """You are a plant disease expert. 
You will respond to the user questions outlining suggested treatments or remedy for the following disease in a plant
{}"""
        prompt = system_prompt.format(predicted_class)
        
        with message_container:
            with st.chat_message("assistant"):
                with st.spinner("Checking for remedy"):
                    message_placeholder = st.empty()
                    full_response = ""
                    llm = get_llm(model, model_provider)
                    for response in llm.stream(prompt):
                        full_response += response.content
                        message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
