import streamlit as st
from streamlit_mermaid import st_mermaid

st.subheader("Plant Disease Identification System (PDIS)", divider = "grey")


st.markdown("""
# About Plant Disease Identification System (PDIS)

Welcome to PDIS, a system designed to help identify plant diseases using image analysis and advanced language models. This application aims to assist farmers, gardeners, and plant enthusiasts in diagnosing plant health issues quickly and efficiently.


## Use Case

PDIS addresses the critical need for rapid and accurate plant disease identification. Early detection is crucial for effective treatment and preventing widespread crop damage. This system provides a user-friendly interface for uploading images of plants, receiving an analysis of the potential disease, and obtaining information on potential remedies.


## Approach

The PDIS application employs a two-pronged approach:

1. **Image-based Disease Detection:** A pre-trained Convolutional Neural Network (CNN), specifically ResNet50, is fine-tuned on a plant disease dataset (sourced from Kaggle). This model analyzes uploaded images to identify potential diseases based on visual patterns.

2. **AI-powered Diagnosis and Remedy Suggestion:**  A large language model (LLM) is integrated to provide detailed diagnostic information and suggestions for treatment based on the predicted disease. The LLM receives the detected disease and plant species as input, generating a human-readable response.  The LLM can be selected from several providers (e.g., Google's Gemini, Ollama models).


## Model Training (Disease Detection)

The disease detection model is trained using a dataset of plant images with associated labels (disease type).  The training process involves:

1. **Data Augmentation:**  Random horizontal flips and rotations are applied to the training images to increase the dataset size and improve model robustness.

2. **Pre-trained Model:**  A pre-trained ResNet50 model is used as a base, leveraging its learned features from a large-scale image dataset (ImageNet).  This significantly reduces training time and improves performance.

3. **Fine-tuning:** Only the final fully connected layers of ResNet50 are unfrozen and trained on the plant disease dataset.  The pre-trained weights of the convolutional layers are kept fixed to preserve the learned features.

4. **Optimization:** The Adam optimizer is used to update the weights of the final layers, minimizing the cross-entropy loss function.

5. **Model Saving:** The trained model is saved for later use in the application, allowing for quick and efficient disease detection.


## Technology Stack

* **Frontend:** Streamlit
* **Backend:** PyTorch, Langchain
* **LLM Providers:** Google AI, Ollama (and potentially others)
* **Image Processing:** PIL (Pillow)
* **Model:** ResNet50


This system continuously learns and improves as more data is added, enhancing its accuracy and reliability in identifying plant diseases.
            """)


st_mermaid("""
graph LR
    A[User] --> B(Upload Image);
    B --> C{ResNet50 Model};
    C --> D[Disease Prediction];
    D --> E{Plant Identification};
    subgraph "LLM Interaction"
        E --> F(Prompt Generation);
        F --> G{LLM (e.g., Gemini, Ollama)};
        G --> H[Diagnosis & Remedy];
    end
    H --> I(Display Results);
    I --> A;
""")

st.markdown("""Explanation of Diagram:

Image Upload (Streamlit): The user uploads an image of a plant through the Streamlit interface.
Disease Detection (ResNet50): The uploaded image is preprocessed and fed into the ResNet50 model, which predicts the disease.
Disease & Plant Identification: The model's output (predicted disease) is combined with a plant type (either predicted or user-specified).
LLM (e.g., Gemini, Ollama): This information is sent as a prompt to the chosen LLM.
Diagnosis & Remedy Suggestion: The LLM processes the prompt and generates a detailed diagnosis and treatment suggestions.
Streamlit Output: The results (disease identification, diagnosis, and remedy) are displayed to the user in the Streamlit application.
""")