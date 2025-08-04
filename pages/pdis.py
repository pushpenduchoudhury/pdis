import os
import uuid
import torch
from PIL import Image
import streamlit as st
from pathlib import Path
import config.conf as conf
from dotenv import load_dotenv
from lib.utils import key_decode
import torchvision.models.resnet
from torchvision import transforms
from langchain.chat_models import init_chat_model
torch.serialization.add_safe_globals([torchvision.models.resnet.ResNet])
load_dotenv()

CSS_FILE = Path(conf.CONFIG_DIR, "pdis_style.css")

with open(CSS_FILE) as f:
    css = f.read()

st.markdown(f'<style>{css}</style>', unsafe_allow_html = True)

os.environ["GOOGLE_API_KEY"] = key_decode(os.getenv("B64_GOOGLE_API_KEY"))
os.environ["GROQ_API_KEY"] = key_decode(os.getenv("B64_GROQ_API_KEY"))

def reset_messages():
    st.session_state.analysis_messages = {"image": ["""<div style="display: flex; font-size: 24px; color: grey; justify-content: center; align-items: center; text-align: center; height: 165px; overflow: hidden;"> <p>Upload an Image to Analyse</p></div>"""],
                                          "analysis" : [],
                                          "diagnosis" : []
                                          }
    st.session_state.disease_detected = False

def clear_messages():
    st.session_state.analysis_messages = {"image": [],
                                          "analysis" : [],
                                          "diagnosis" : []
                                          }
    st.session_state.disease_detected = False

st.set_page_config(
    page_title = "PDIS",
    page_icon = "🌱",
    layout = "wide",
    initial_sidebar_state = "collapsed"
)

st.markdown(
    f"""
<style>
    .st-emotion-cache-595tnf:before {{
        content: "𖡎 PDIS";
        font-weight: bold;
        font-size: xx-large;
    }}
</style>""",
        unsafe_allow_html=True,
    )


# Set Title
title_col = st.columns([0.8, 12, 0.5])
with title_col[0]:
    st.image(image = Path(conf.ASSETS_DIR, "pdis.png"), use_container_width = True)
with title_col[1]:
    st.header("Plant Disease Identification System", divider = "rainbow", anchor = False)
with title_col[2]:
    st.write("")
    st.button(":material/mop:", on_click = lambda: reset_messages(), type = "secondary", use_container_width = True)


def get_ollama_models() -> list:
    import ollama
    ollama_client = ollama.Client()
    llm = ollama_client.list()
    model_list = [i['model'] for i in llm['models']]
    return model_list

available_models = conf.AVAILABLE_MODELS

input_col, output_col = st.columns([0.3, 0.7])

if "analysis_messages" not in st.session_state:
    st.session_state.analysis_messages = {"image": ["""<div style="display: flex; font-size: 24px; color: grey; justify-content: center; align-items: center; height: 165px; overflow: hidden;"> <p>Upload an Image to Analyse</p></div>"""],
                                          "analysis" : [],
                                          "diagnosis" : []
                                          }

if "disease_detected" not in st.session_state:
    st.session_state.disease_detected = False

if "plant" not in st.session_state:
    st.session_state.plant = None

if "disease" not in st.session_state:
    st.session_state.disease = None

# disease_detected = False
uploaded_file = input_col.file_uploader(
                                        "Upload an image of a Plant", 
                                        accept_multiple_files = False, 
                                        type = ["jpg", "jpeg"], 
                                        label_visibility = "collapsed")

col1, col2, col3 = input_col.columns([0.3, 0.3, 0.4])

with input_col.expander(":grey[⚙️ Model Config]"):
    st.markdown("#####  :grey[Disease Detection Model]")
    st.selectbox("Disease Detection Model", options = os.listdir(conf.MODEL_DIR), label_visibility = "collapsed")
    st.markdown(':grey[Model Accuracy:]<br><hr>', unsafe_allow_html = True)
    
    col4, col5 = st.columns([0.2, 0.8])
    col4.markdown("#####  :grey[LLM:]")
    model_provider = col5.radio("Model Provider", options = available_models.keys(), horizontal = True, label_visibility = "collapsed")

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



image_preview_col, diagnosis_col = output_col.columns([0.2, 0.8])
image_container = image_preview_col.container(height = 185, border = True)
analysis_container = diagnosis_col.container(height = 185, border = True)
diagnosis_container = output_col.container(height = 315, border = True)
analysis_container.subheader(":blue[Analysis]", divider = "red" if st.session_state.disease_detected else "grey", anchor = False)
diagnosis_container.subheader(":blue[Diagnosis]", divider = "grey", anchor = False)

if uploaded_file is not None:
    st.session_state.analysis_messages["image"] = []
    st.session_state.analysis_messages["image"].append(uploaded_file)

if uploaded_file is None:
    reset_messages()
    st.session_state.disease_detected = False


for message in st.session_state.analysis_messages["image"]:
    if len(st.session_state.analysis_messages["image"]) == 1 and uploaded_file is None:
        image_container.markdown(message, unsafe_allow_html = True)
    else:
        image_container.image(message, width = 150,)

def disease_class_parser(predicted_class):
    plant = predicted_class.split("___")[0] if "___" in predicted_class else predicted_class.split("__")[0] if "__" in predicted_class else predicted_class.split("_")[0]
    disease = predicted_class.split("___")[1] if "___" in predicted_class else predicted_class.split("__")[1] if "__" in predicted_class else predicted_class.split("_")[1]
    return plant, disease

def analyze():
    st.session_state.analysis_messages["analysis"] = []
    st.session_state.analysis_messages["diagnosis"] = []
    with analysis_container:
        with st.spinner("Analyzing..."):
            predicted_class = detect_disease(image = uploaded_file)
            st.session_state.plant, st.session_state.disease = disease_class_parser(predicted_class)
        
    if "healthy" in st.session_state.disease:
        st.session_state.analysis_messages["analysis"].append(f'<span style="font-size: 20px; color: grey;"> Plant:</span> <span style="font-size: 22px;">{st.session_state.plant}</span> <br> <span style="font-size: 20px; color: grey;">Condition:</span> <span style="font-size: 25px; color: green;">**Healthy**</span>')
        st.session_state.disease_detected = False
        st.session_state.analysis_messages["diagnosis"].append('<span style="display: block; text-align:center;color: grey">(No diagnosis required)</span>')
    else:
        st.session_state.analysis_messages["analysis"].append(f'<span style="font-size: 20px; color: grey;"> Plant:</span> <span style="font-size: 22px;">{st.session_state.plant}</span> <br> <span style="font-size: 20px; color: grey;">Condition:</span> <span style="font-size: 25px; color: red;">**{st.session_state.disease.title()}** detected...!</span>')
        st.session_state.disease_detected = True
        st.session_state.analysis_messages["diagnosis"].append('<span style="display: block; text-align:center;color: grey">(Click on "Diagnose" button for more details on the disease)</span>')
        

def diagnose():
    st.session_state.analysis_messages["diagnosis"] = []
    prompt = conf.SYSTEM_PROMPT.format(plant = st.session_state.plant, disease = st.session_state.disease)
    with diagnosis_container:
        try:
            with st.spinner("Diagnosing disease..."):
                message_placeholder = st.empty()
                full_response = ""
                llm = get_llm(model, model_provider)
                for response in llm.stream(prompt):
                    full_response += response.content
                    message_placeholder.markdown(full_response + "▌")
        except Exception as e:
            full_response = str(f":red[{e}]")
        # message_placeholder.markdown(full_response)
        st.session_state.analysis_messages["diagnosis"].append(full_response)
        

analyze_button = col1.button("Analyze", use_container_width = True, disabled = False if uploaded_file is not None else True, type = "secondary")
if analyze_button:
    analyze()
    # st.rerun()
    
diagnose_button = col2.button("Diagnose", use_container_width = True, disabled = not st.session_state.disease_detected, type = "primary")
if diagnose_button:
    diagnose()
    # st.rerun()


auto_diagnose = col3.toggle("Auto Diagnose")
if auto_diagnose:
    if len(st.session_state.analysis_messages["diagnosis"]) == 0:
        if uploaded_file:
            analyze()
            for message in st.session_state.analysis_messages["analysis"]:
                analysis_container.markdown(message, unsafe_allow_html = True)
            diagnose()
            # for message in st.session_state.analysis_messages["diagnosis"]:    
            #     diagnosis_container.markdown(message, unsafe_allow_html = True)
            # st.rerun()
    elif uploaded_file:
        analyze()
        for message in st.session_state.analysis_messages["analysis"]:
            analysis_container.markdown(message, unsafe_allow_html = True)
        diagnose()
        
for message in st.session_state.analysis_messages["analysis"]:
    analysis_container.markdown(message, unsafe_allow_html = True)

for message in st.session_state.analysis_messages["diagnosis"]:    
    diagnosis_container.markdown(message, unsafe_allow_html = True)

with st.sidebar:
    with st.expander("𓆣 Debugging"):
        st.write(st.session_state.analysis_messages)