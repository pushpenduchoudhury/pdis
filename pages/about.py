import streamlit as st
import streamlit.components.v1 as components
import urllib.parse
from streamlit_mermaid import st_mermaid


# Page setup
st.set_page_config(
    page_title="About - PDIS",
    page_icon="🌱",
    layout="wide"
)

# Title
st.header("🌿 Plant Disease Identification System", divider = 'grey')

# Use Case
st.subheader("🧩 Use Case")
st.markdown("""
In agriculture, timely detection of plant diseases is crucial to prevent crop loss and maintain food quality. Traditional methods are time-consuming and require expert knowledge.  
**PDIS (Plant Disease Identification System)** is an AI-powered application that allows users to upload images of plant leaves and get real-time disease analysis, diagnosis, and treatment suggestions using deep learning and LLMs.
""")

# Approach
st.subheader("🔬 Approach")
st.markdown("""
**PDIS** integrates:
- 🎯 **Computer Vision** to identify plant diseases from images.
- 🧠 **Large Language Models (LLMs)** to provide detailed disease explanations and care suggestions.
- 🖥️ **Streamlit frontend** for a user-friendly interface.

**Workflow:**
1. Users upload plant images via a web UI.
2. A deep learning model (ResNet50) classifies the plant and detects disease.
3. A large language model generates a natural language diagnosis and recommendation.
4. Users can select different model providers (Groq, Google Gemini, Ollama).
""")

# Model Training
st.subheader("🧠 Model Training")
st.markdown("""
The disease detection model was trained using the [PlantVillage dataset](https://www.kaggle.com/datasets/emmarex/plantdisease) from Kaggle, which includes over 54,000 labeled images of healthy and diseased leaves.

**Training Highlights:**
- 📚 **Dataset**: Multi-class folder structure
- 🖼 **Augmentation**: Resize, horizontal flip, rotation, normalization
- 🧱 **Model**: Pretrained ResNet-50
- 🔍 **Loss**: CrossEntropyLoss
- ⚙️ **Optimizer**: Adam
- 💾 **Saved as**: `plant_disease_full_model.pth`

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


st.subheader("⚙️ Technologies Used")
st.markdown("""

Frontend: Streamlit

Image Processing: PIL, torchvision.transforms

Model Training: PyTorch, ResNet-50

LLM Integration: LangChain, Groq, Gemini, Ollama

Utility: TQDM, dotenv, Pathlib
""")


st.subheader("🧭 Architecture Diagram")
st.markdown("Below is the architecture of the Plant Disease Identification System:")

# Mermaid diagram as image
st_mermaid('''
graph TD
A[🌐 Streamlit Frontend<br>• Image Upload UI<br>• Config & Analysis View<br>• Diagnose Button] --> B[🧠 Inference Engine<br>• Image Preprocessing<br>• Load ResNet50 Model<br>• Predict Disease]
B --> C[🔗 LangChain<br>• Prompt Construction<br>• Streamed Responses]
C --> D1[🤖 Groq<br>LLM: LLaMA, Mistral, etc.]
C --> D2[🤖 Google Gemini<br>LLM: Gemini 1.5 Flash]
C --> D3[🤖 Ollama<br>LLM: Local models]
A --> E[🛠 Model Selector UI<br>• Choose LLM Provider<br>• Select Model]
''')

# Encode for Mermaid.ink
# encoded = urllib.parse.quote(mermaid_code)
# mermaid_img_url = f"https://mermaid.ink/img/{encoded}"

# # Display diagram
# st.image(mermaid_img_url, caption="PDIS System Architecture (Mermaid)", use_container_width=True)



# Final Notes
st.subheader("👩‍🌾 Final Notes")
st.markdown("""

🧪 More plant types and diseases can be added to the dataset to increase scope.

🔄 The current system can be enhanced with multilingual support.

📈 Higher accuracy can be achieved with more training epochs and improved datasets.

☁️ PDIS can be deployed to cloud platforms for field-level integration.

<center>2025 Plant Disease Identification System (PDIS)</center> """, unsafe_allow_html=True)