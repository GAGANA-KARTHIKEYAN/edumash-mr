# 🚀 eduMASH-MR: Deployment Guide

This guide describes how to deploy your **eduMASH-MR** AI Tutor system to the web using **Streamlit Community Cloud**.

---

## 🏗️ Phase 1: Prepare for GitHub

Streamlit Community Cloud deploys directly from a GitHub repository. 

1. **Initialize Git** (if not already done):
   ```bash
   git init
   git add .
   git commit -m "Initial commit for eduMASH-MR"
   ```

2. **Handle Large Files (Important):**
   If your `weights/` directory contains large files (over 50MB), you MUST use **Git LFS**:
   ```bash
   git lfs install
   git lfs track "weights/*.pt"
   git add .gitattributes
   git commit -m "Add git elements for large weights"
   ```

3. **Push to GitHub:**
   - Create a new repository on [GitHub](https://github.com/new).
   - Link and push your code:
     ```bash
     git remote add origin https://github.com/YOUR_USERNAME/edumash-mr.git
     git branch -M main
     git push -u origin main
     ```

---

## ☁️ Phase 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io).
2. Click **"New app"**.
3. Select your repository, the **main** branch, and set the Main file path to `app.py`.
4. **Advanced Settings (Secrets):**
   - You can pre-configure your API keys so they are always ready.
   - Go to **Settings -> Secrets** and paste:
     ```toml
     GROQ_API_KEY = "your_groq_key_here"
     GEMINI_API_KEY = "your_gemini_key_here"
     ```
5. Click **Deploy!**

---

## 🛠️ Phase 3: Technical Integrity

The project includes two critical configuration files that I have already created for you:

1.  **`packages.txt`**: Tells Streamlit Cloud to install `tesseract-ocr` and `ffmpeg` at the OS level so your image and voice features work in the cloud.
2.  **`requirements.txt`**: Lists all Python libraries needed for the RAG and Graph engines.

---

## ⚠️ Common Cloud Troubleshooting

- **"ModuleNotFoundError: torchvision"**: This is fixed in the new `requirements.txt`.
- **Memory Limits:** Streamlit Cloud has a 1GB RAM limit. If your knowledge graph becomes extremely large (>500 nodes), the app may restart. 
- **Cold Starts:** On the first run, downloading the embedding models (`sentence-transformers`) might take 2-3 minutes. Subsequent runs will be fast.
