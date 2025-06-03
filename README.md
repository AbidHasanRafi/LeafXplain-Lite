# 🌿 LeafXplain-Lite: Advanced Plant Disease Diagnosis

![App Header](https://raw.githubusercontent.com/AbidHasanRafi/LeafXplain-Lite/main/assets/app_header.png)

**LeafXplain-Lite** is a smart, explainable AI-based web application for real-time plant disease detection. It provides intuitive visualizations, interpretable model outputs, and expert treatment suggestions through Gemini AI — all in one lightweight package.


## Features at a Glance

### Image-Based Disease Analysis
- Upload a leaf image to get a fast and accurate disease diagnosis.
- Powered by a CNN model with attention mechanisms and Grad-CAM explanations.
- Visualizes which part of the leaf influenced the model's decision.


## App Walkthrough

### Diagnosis Overview – Instant Summary

When an image is uploaded, the app provides a quick summary of predictions, including:
- predicted disease
- Grad-Cam Visuals
- Predicted Result

![Diagnosis Overview](https://raw.githubusercontent.com/AbidHasanRafi/LeafXplain-Lite/main/assets/diagonosis_overview.png)


### Detailed Analysis – Interpretability & Insights

This section breaks down the prediction with clear, scientific visualizations:
- **Original Image**: What you uploaded.
- **Attention Heatmap**: Shows what the model is focusing on.
- **Class Activation Map (CAM)**: Highlights influential image regions.
- **Prediction Distribution**: Probability scores for each class.

![Detailed Analysis](https://raw.githubusercontent.com/AbidHasanRafi/LeafXplain-Lite/main/assets/detailed_analysis.png)


### Diagnosis Result

Once a prediction is made, LeafXplain-Lite provides the diagonosis results:
- Predicted Disease
- Disease Confidence
- Top Predictions
- Processing Time
- Model Size

![Diagnosis Result](https://raw.githubusercontent.com/AbidHasanRafi/LeafXplain-Lite/main/assets/diagonosis_result.png)


## Live Detection from Camera

- Switch to **Live Camera Mode** for real-time disease detection.
- Get instant visual feedback with bounding boxes and overlays.
- Gemini-generated advice is just one click away.


## Quick Start

### Prerequisites
- Python 3.8+
- Streamlit
- PyTorch
- Google Generative AI SDK

### Installation
```bash
git clone https://github.com/AbidHasanRafi/LeafXplain-Lite.git
cd LeafXplain-Lite
pip install -r requirements.txt
````

### Running the App

```bash
streamlit run app.py
```


## Configuration Guide

### 1. Add Gemini API Key

* Get your API key from Google
* Add to Streamlit secrets or directly in code (for testing)

### 2. Load or Upload Model

* Use the default provided model or upload your own `.pth`
* Model switching is available via dropdown


## Model Details

### Architecture

```python
class PlantDiseaseCNN(nn.Module):
    def __init__(self, num_classes):
        super(PlantDiseaseCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # Additional layers...
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
```


## Supported Plant Diseases

Supports detection of 20 common plant diseases, including:

* Apple Scab
* Tomato Late Blight
* Grape Black Rot
* Corn Common Rust
* Potato Early Blight
  ... and more.

## Why Use LeafXplain-Lite?

* **Explainable AI** — Grad-CAM shows where the model focuses
* **Real-time Detection** — From image uploads or live camera
* **Gemini-Powered Expertise** — Instant, actionable advice
* **Lightweight** — Small model size, fast on CPU or GPU
* **Clean UI** — Tab-based view with dark/light mode support

## Contact

- **Project Maintainer** – [Md. Abid Hasan Rafi](mailto:ahr16.abidhasanrafi@gmail.com)
- **Project Contributor** – [Pankaj Bhowmik](mailto:pankaj@hstu.ac.bd)
- **GitHub Repository** – [https://github.com/abidhasanrafi/leafxplain-lite](https://github.com/abidhasanrafi/leafxplain-lite)