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
- **Saved as**: `plant_disease_full_model.pth`

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
""")


st.markdown("""##### :grey[Model Training Diagram]

Below is the process for training model for Plant Disease Identification System:""")

# Mermaid diagram as image

st_mermaid('''graph TD
    A[Start] --> B[Load Dataset from Folder]
    B --> C[Apply Transforms<br>Resize, Flip, Rotate, Normalize]
    C --> D[ImageFolder Dataset]
    D --> E[DataLoader with Batch Size 8]
    E --> F[Load Pretrained ResNet50]
    F --> G[Freeze All Layers]
    G --> H[Replace Final FC Layer<br>Linear -> ReLU -> Dropout -> Linear]

    H --> I[Move Model to Device CPU/GPU]
    I --> J[Define Loss Function - CrossEntropyLoss]
    J --> K[Define Optimizer - Adam on FC params]

    K --> L[Training Loop]
    L --> M[For each batch:<br>Forward Pass -> Loss -> Backward -> Optimizer Step]
    M --> N[Calculate Loss and Accuracy]

    N --> O[Save Trained Model to .pth]

    O --> P[Load Trained Model - Optional]
    P --> Q[Model.eval for Inference]

    Q --> R[Preprocess Image<br>Resize, Normalize, ToTensor]
    R --> S[Run Inference<br>model - input_tensor]
    S --> T[Get Predicted Class]

    T --> V[End]
''', zoom = False, pan = False)


subheader("Technologies Used")
st.markdown("""

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