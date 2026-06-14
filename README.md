# Bird 20-Species Image Classification App

An end-to-end lightweight computer vision web application built for **Streamlit Cloud Deployment**.

## Features
- **Auto-Ingestion Pipeline:** Uses `kagglehub` to directly load target images onto cloud runtimes.
- **Efficient Architecture:** Powered by a lightweight MobileNetV2 architecture in PyTorch (Tensorflow is bypassed to prevent resource bloating).
- **Interactive UI:** Users can easily drag and drop localized test instances via their web browser.

## Deployment Instructions
1. Upload these files to a public GitHub repository.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Connect your repository and set the main file path to `streamlit_app.py`.
4. Click **Deploy!**