# inference/demo_app.py

import sys
import os
import streamlit as st
from PIL import Image
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vision.model import build_model
from vision.feature_extractor import extract_single_embedding
from multimodal.multimodal_model import MultimodalModel
from llm.embedding_extractor import get_text_embedding

# ---------------------------
# PAGE CONFIG + THEME
# ---------------------------
st.set_page_config(page_title="EO Multimodal AI", layout="wide")

st.markdown("""
<style>
body {
    background-color: #0e1a2b;
    color: white;
}
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}
h1, h2, h3 {
    color: #00d4ff;
}
.chat-box {
    background-color: #1b2a3a;
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# SIDEBAR NAV
# ---------------------------
page = st.sidebar.radio("Navigation", ["🏠 Home", "🤖 Chatbot"])

# ---------------------------
# HOME PAGE
# ---------------------------
if page == "🏠 Home":

    
    import time

    title_text = "🌍 EO Multimodal Intelligence System"
    title_placeholder = st.empty()

    typed_text = ""
    for char in title_text:
        typed_text += char
        title_placeholder.markdown(f"# {typed_text}")
        time.sleep(0.03)  # speed (reduce to 0.01 for faster)

    st.markdown("""
### 🛰️ Satellite Intelligence System

This project combines:

- Vision Transformer (Satellite Image Understanding)
- Text Embeddings (Semantic Meaning)
- Multimodal Fusion (Decision Making)

### 🚀 Features
- Land classification
- Confidence estimation
- Intelligent chatbot reasoning

### 🌱 Applications
- Agriculture
- Smart cities
- Environmental monitoring
- Disaster analysis
""")

# ---------------------------
# CHATBOT PAGE
# ---------------------------
if page == "🤖 Chatbot":

    st.title("🤖 EO Multimodal Chatbot")

    if "chat" not in st.session_state:
        st.session_state.chat = []

    col1, col2 = st.columns([1, 2])

    with col1:
        uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

    with col2:
        user_input = st.text_input("Ask a question")

    # ---------------------------
    # DEFAULT QUESTIONS
    # ---------------------------
    st.markdown("### 💡 Try these questions:")
    sample_qs = [
        "What is the land type?",
        "Show all class probabilities",
        "Which are top 3 predictions?",
        "Explain the prediction",
        "What is the confidence level?"
    ]

    cols = st.columns(len(sample_qs))
    for i, q in enumerate(sample_qs):
        if cols[i].button(q):
            user_input = q

    # ---------------------------
    # PROCESS
    # ---------------------------
    if uploaded_file and user_input:

        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load Vision Model
        vision_model = build_model().to(device)
        v_ckpt = torch.load("checkpoints/vision/best_model.pth", map_location=device)
        vision_model.load_state_dict(v_ckpt["model_state_dict"])
        vision_model.eval()

        # Load Fusion Model
        fusion_model = MultimodalModel().to(device)
        f_ckpt = torch.load("checkpoints/fusion_best.pth", map_location=device)
        fusion_model.load_state_dict(f_ckpt["model_state_dict"])
        fusion_model.eval()

        # Extract embedding
        eo_embedding = extract_single_embedding(image, vision_model, device)
        eo_embedding = torch.tensor(eo_embedding).to(device).float()

        # Classes
        class_names = [
            "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
            "Industrial", "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"
        ]

        # Text embeddings
        class_embeddings = []
        for cls in class_names:
            text = f"This is a satellite image showing {cls.lower()} land cover."
            class_embeddings.append(get_text_embedding(text))

        class_embeddings = torch.stack(class_embeddings).to(device)

        # ---------------------------
        # GET ALL SCORES
        # ---------------------------
        scores = []

        with torch.no_grad():
            for i in range(len(class_embeddings)):
                v = eo_embedding.unsqueeze(0)
                t = class_embeddings[i].unsqueeze(0)
                out = fusion_model(v, t)
                prob = torch.softmax(out, dim=1)[0][i].item()
                scores.append(prob)

        scores = np.array(scores)
        sorted_idx = np.argsort(scores)[::-1]

        pred_class = class_names[sorted_idx[0]]
        confidence = scores[sorted_idx[0]]

        # ---------------------------
        # BETTER CHATBOT LOGIC
        # ---------------------------
        def chatbot(q):

            q = q.lower()

            if "land type" in q:
                return f"🌿 The image is classified as **{pred_class}** with {confidence*100:.2f}% confidence."

            elif "probabilities" in q:
                result = "📊 Class probabilities:\n"
                for i in sorted_idx:
                    result += f"- {class_names[i]} → {scores[i]*100:.2f}%\n"
                return result

            elif "top" in q:
                result = "🏆 Top 3 predictions:\n"
                for i in sorted_idx[:3]:
                    result += f"- {class_names[i]} ({scores[i]*100:.2f}%)\n"
                return result

            elif "confidence" in q:
                return f"🎯 Confidence: {confidence*100:.2f}% for {pred_class}"

            elif "difference" in q:
                diff = scores[sorted_idx[0]] - scores[sorted_idx[1]]
                return f"📉 Difference between top two classes is {diff*100:.2f}%."
            elif "how" in q or "classify" in q:
                return f"""
🧠 **How the model classifies the image:**

1. 🛰️ **Vision Model**
   - Extracts visual patterns like vegetation, buildings, water
   - Converts image into a numerical feature vector

2. 🧠 **Text Embeddings**
   - Each land type (Forest, River, etc.) is converted into a semantic vector

3. 🔗 **Fusion Model**
   - Combines image features + text embeddings
   - Computes similarity scores for each class

4. 📊 **Decision**
   - The class with highest similarity → **{pred_class}**
   - Confidence → {confidence*100:.2f}%

This is how the system intelligently predicts land type.
"""

            elif "explain" in q or "why" in q:
                return f"""
🧠 **Explanation**

The model predicts **{pred_class}** because:
- Visual patterns match {pred_class.lower()}
- Texture and spatial layout align with this class
- Feature similarity is highest compared to others
"""
            else:
                return f"🤖 Prediction: {pred_class} ({confidence*100:.2f}%)"

        response = chatbot(user_input)

        # Save chat
        st.session_state.chat.append(("You", user_input))
        st.session_state.chat.append(("Bot", response))

    # ---------------------------
    # DISPLAY CHAT
    # ---------------------------
    st.markdown("### 💬 Conversation")

    for role, msg in st.session_state.chat:
        st.markdown(f"<div class='chat-box'><b>{role}:</b> {msg}</div>", unsafe_allow_html=True)