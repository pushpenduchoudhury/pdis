import streamlit as st
from pathlib import Path
import config.conf as conf
from streamlit_mermaid import st_mermaid

CSS_FILE = Path(conf.CONFIG_DIR, "about_style.css")
with open(CSS_FILE) as f:
    css = f.read()
st.markdown(f'<style>{css}</style>', unsafe_allow_html = True)

# Page setup
st.set_page_config(
    page_title="About - PDIS",
    page_icon="🌿",
    layout="wide"
)

# Title
st.header("🌿 :rainbow[Plant Disease Identification System]", divider = 'rainbow')

def subheader(text):
    st.subheader(f":blue[{text}]", divider = "grey", anchor = False)

# Use Case
subheader("Use Case")
st.markdown("""In agriculture, timely detection of plant diseases is crucial to prevent crop loss and maintain food quality. Traditional methods are time-consuming and require expert knowledge.  
**PDIS (Plant Disease Identification System)** is an AI-powered application that allows users to upload images of plant leaves and get real-time disease analysis, diagnosis, and treatment suggestions using deep learning and LLMs.
""")

# Approach
subheader("Approach")
st.markdown("""
##### :grey[**PDIS** integrates:]
- **Computer Vision** to identify plant diseases from images.
- **Large Language Models (LLMs)** to provide detailed disease explanations and care suggestions.
- **Streamlit frontend** for a user-friendly interface.


##### :grey[**Workflow:**]

Below is the workflow architecture of the Plant Disease Identification System:
""")

st_mermaid("""
graph TD
    A[<b>User</b><br>Uploads Plant Image] --> B[<b>Streamlit Web UI</b>]
    B --> C[<b>ResNet50 Model</b><br>Classifies Plant & Detects Disease]
    C --> D[<b>Generate Prompt</b><br>for Disease Diagnosis]
    D --> E[<b>LangChain LLM Interface</b>]
    E --> F1[<b>Groq</b><br>LLaMA / Mistral]
    E --> F2[<b>Google Gemini</b><br>Gemini 1.5 Flash]
    E --> F3[<b>Ollama</b><br>Local Models]
    F1 --> G[<b>Natural Language Diagnosis</b>]
    F2 --> G
    F3 --> G
    """, zoom = False, pan = False)


# Model Training
subheader("Model Training")
st.markdown("""
The disease detection model was trained using the [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset/) from Kaggle, which includes over 87,000 RGB images of healthy and diseased crop leaves which is categorized into 38 different classes.

##### :grey[**Training Highlights:**]
- **Dataset**: Multi-class folder structure
- **Augmentation**: Resize, horizontal flip, rotation, normalization
- **Model**: Pretrained ResNet-50
- **Loss**: CrossEntropyLoss
- **Optimizer**: Adam

```python
model = models.resnet50(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 1024),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(1024, num_classes)
)
```

##### :grey[**Steps for Training:**]

1. Data Preparation
    * Images are organized in folders by class.
    * Data is augmented with resizing, random flips, and rotations.
    * Images are normalized to match pretrained model expectations.

2. Model Setup
    * A ResNet50 model pretrained on ImageNet is used.
    * All original layers are frozen to retain learned features.
    * The final classification layer is replaced to match the number of target classes.

3. Training Configuration
    * Only the new classifier head is trained; the rest of the model remains unchanged.
    * Cross-entropy loss is used for multi-class classification.
    * The Adam optimizer updates the classifier's weights.

4. Data Preparation
    * The model is trained for a set number of epochs.
    * For each batch:
        * Images and labels are loaded and processed.
        * The model predicts class probabilities.
        * Loss and accuracy are calculated.
        * The classifier's weights are updated based on the loss.
    * After each epoch, overall training loss and accuracy are reported.

""", unsafe_allow_html = True)


st.markdown("""##### :grey[Model Training Diagram]

Below is the process for training model for Plant Disease Identification System:""")

# Mermaid diagram as image

st_mermaid("""flowchart TD
    A[Start Training]
    B[Train for N Epochs]

    subgraph Epoch Loop
        C[For each batch]
        D[Load & process images and labels]
        E[Model predicts class probabilities]
        F[Calculate loss and accuracy]
        G[Update classifier weights based on loss]
    end

    H[After each epoch: Report training loss and accuracy]
    I[End Training]

    %% Connections
    A --> B --> C
    C --> D --> E --> F --> G
    G --> C
    C --> H
    H --> B
    B --> I
""", zoom = False, pan = False)


subheader("Technologies Used")
st.markdown(f"""

:grey[Frontend:] Streamlit

:grey[Image Processing:] PIL | torchvision.transforms

:grey[Model Training:] PyTorch | ResNet-50

:grey[LLM Integration:] LangChain | Groq | Gemini | Ollama

:grey[Utility:] dotenv | Pathlib
""")

# Final Notes
subheader("Final Notes")
st.markdown("""

> Higher accuracy can be achieved with more training epochs and improved hardware support.

> More plant types and diseases can be added to the dataset to increase scope.

> PDIS can be deployed to cloud platforms for field-level integration.

<br>
<br>
<center>@2025 Plant Disease Identification System (PDIS)</center>""", unsafe_allow_html = True)


