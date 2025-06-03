import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import warnings
import time
import pandas as pd
import seaborn as sns
from io import BytesIO
import google.generativeai as genai
import tempfile
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

warnings.filterwarnings('ignore')

# Configure Gemini API
GEMINI_API_KEY = "Enter_Your_Gemini_API_Key_Here"  # Replace with your actual API key
genai.configure(api_key=GEMINI_API_KEY)

# Initialize Gemini model
generation_config = {
    "temperature": 0.5,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 4096,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

gemini_model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    generation_config=generation_config,
    safety_settings=safety_settings
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model architecture
class PlantDiseaseCNN(nn.Module):
    def __init__(self, num_classes):
        super(PlantDiseaseCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32, bias=False),
            nn.Conv2d(32, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64, bias=False),
            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        features = self.features[:-1](x)
        attention = self.features[-1](features)
        attended = features * attention
        return self.classifier(attended)

# class names with detailed descriptions
CLASS_NAMES = [
    'Apple_Apple Scab', 'Apple_Black Rot', 'Apple_Cedar Apple Rust',
    'Bell Pepper_Bacterial Spot', 'Cherry_Powdery Mildew',
    'Corn (Maize)_Cercospora Leaf Spot', 'Corn (Maize)_Common Rust',
    'Corn (Maize)_Northern Leaf Blight', 'Grape_Black Rot',
    'Grape_Esca (Black Measles)', 'Grape_Leaf Blight',
    'Peach_Bacterial Spot', 'Potato_Early Blight', 'Potato_Late Blight',
    'Strawberry_Leaf Scorch', 'Tomato_Bacterial Spot',
    'Tomato_Early Blight', 'Tomato_Late Blight',
    'Tomato_Septoria Leaf Spot', 'Tomato_Yellow Leaf Curl Virus'
]

DISEASE_DESCRIPTIONS = {
    'Apple_Apple Scab': 'Fungal disease causing dark, scaly lesions on leaves and fruit.',
    'Apple_Black Rot': 'Fungal disease causing brown rot with concentric rings on fruit.',
    'Apple_Cedar Apple Rust': 'Fungal disease with bright orange spots on leaves.',
    'Bell Pepper_Bacterial Spot': 'Bacterial disease causing small, water-soaked leaf spots.',
    'Cherry_Powdery Mildew': 'Fungal disease with white powdery growth on leaves.',
    'Corn (Maize)_Cercospora Leaf Spot': 'Fungal disease causing small, circular leaf spots with tan centers.',
    'Corn (Maize)_Common Rust': 'Fungal disease with small, reddish-brown pustules on leaves.',
    'Corn (Maize)_Northern Leaf Blight': 'Fungal disease causing long, elliptical gray-green lesions.',
    'Grape_Black Rot': 'Fungal disease causing brown leaf spots with black fruiting bodies.',
    'Grape_Esca (Black Measles)': 'Complex disease causing tiger-stripe patterns on leaves.',
    'Grape_Leaf Blight': 'Bacterial disease causing angular leaf spots with yellow halos.',
    'Peach_Bacterial Spot': 'Bacterial disease causing small, purple-black leaf spots.',
    'Potato_Early Blight': 'Fungal disease causing concentric rings on leaves resembling targets.',
    'Potato_Late Blight': 'Famous for Irish Potato Famine, causes water-soaked leaf lesions.',
    'Strawberry_Leaf Scorch': 'Fungal disease causing purple spots that turn brown and scorched.',
    'Tomato_Bacterial Spot': 'Bacterial disease causing small, dark leaf spots with yellow halos.',
    'Tomato_Early Blight': 'Fungal disease causing target-like lesions on older leaves.',
    'Tomato_Late Blight': 'Destructive disease causing large, water-soaked leaf lesions.',
    'Tomato_Septoria Leaf Spot': 'Fungal disease causing small, circular spots with gray centers.',
    'Tomato_Yellow Leaf Curl Virus': 'Viral disease causing yellowing and upward curling of leaves.'
}

# load model function
@st.cache_resource
def load_model(model_path):
    try:
        model = PlantDiseaseCNN(num_classes=len(CLASS_NAMES)).to(device)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# grad-CAM implementation
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # register hooks
        target_layer.register_forward_hook(self.save_activations)
        target_layer.register_full_backward_hook(self.save_gradients)
    
    def save_activations(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def __call__(self, x, class_idx=None):
        # forward pass
        output = self.model(x)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
        
        # zero gradients
        self.model.zero_grad()
        
        # backward pass for specific class
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1
        output.backward(gradient=one_hot)
        
        # get pooled gradients
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # weight the activations
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]
        
        # generate heatmap
        heatmap = torch.mean(self.activations, dim=1).squeeze()
        heatmap = torch.maximum(heatmap, torch.zeros_like(heatmap))
        heatmap /= torch.max(heatmap)
        
        return heatmap.cpu().numpy()

def create_combined_figure(original_image, heatmap, alpha=0.5):
    """Create a responsive combined figure with original and heatmap"""
    # convert to numpy arrays
    img_array = np.array(original_image)
    
    # resize heatmap to match image dimensions
    heatmap_resized = np.array(Image.fromarray(heatmap).resize(original_image.size, Image.BILINEAR))
    
    # normalize heatmap
    heatmap_normalized = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min())
    
    # create figure with dark mode compatibility
    plt.style.use('dark_background' if st.get_option("theme.base") == "dark" else 'default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'wspace': 0.05})
    
    # display original image
    ax1.imshow(img_array)
    ax1.set_title("Original Image", fontsize=12, pad=10, fontweight='bold', 
                 color='white' if st.get_option("theme.base") == "dark" else 'black')
    ax1.axis('off')
    
    # display original image with heatmap overlay
    ax2.imshow(img_array)
    heatmap_display = ax2.imshow(heatmap_normalized, cmap='inferno', alpha=alpha)
    ax2.set_title("Model Attention", fontsize=12, pad=10, fontweight='bold',
                 color='white' if st.get_option("theme.base") == "dark" else 'black')
    ax2.axis('off')
    
    # add colorbar with smaller size
    cbar = fig.colorbar(heatmap_display, ax=ax2, fraction=0.046, pad=0.01)
    cbar.ax.tick_params(labelsize=8, colors='white' if st.get_option("theme.base") == "dark" else 'black')
    
    plt.tight_layout()
    return fig

def create_detailed_analysis_figure(image, heatmap, predictions):
    """Create a comprehensive analysis figure with multiple visualizations"""
    plt.style.use('dark_background' if st.get_option("theme.base") == "dark" else 'default')
    fig = plt.figure(figsize=(16, 12))
    
    # Grid layout
    gs = fig.add_gridspec(3, 3)
    
    # Original Image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(np.array(image))
    ax1.set_title("Original Image", fontsize=10)
    ax1.axis('off')
    
    # Heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(np.array(image))
    heatmap_resized = np.array(Image.fromarray(heatmap).resize(image.size, Image.BILINEAR))
    heatmap_normalized = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min())
    ax2.imshow(heatmap_normalized, cmap='inferno', alpha=0.5)
    ax2.set_title("Attention Heatmap", fontsize=10)
    ax2.axis('off')
    
    # Class Activation Map
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(heatmap_normalized, cmap='viridis')
    ax3.set_title("Class Activation Map", fontsize=10)
    ax3.axis('off')
    
    # Prediction Distribution
    ax4 = fig.add_subplot(gs[1, :])
    probs = predictions.squeeze().cpu().numpy()
    top_indices = np.argsort(probs)[-5:][::-1]
    top_classes = [CLASS_NAMES[i] for i in top_indices]
    top_probs = [probs[i] for i in top_indices]
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_indices)))
    ax4.barh(top_classes, top_probs, color=colors)
    ax4.set_xlabel("Probability", fontsize=10)
    ax4.set_title("Top Predictions", fontsize=12)
    ax4.set_xlim(0, 1)
    
    # Confidence Distribution
    ax5 = fig.add_subplot(gs[2, :])
    sns.kdeplot(probs, ax=ax5, fill=True, color='skyblue')
    ax5.set_xlabel("Probability", fontsize=10)
    ax5.set_ylabel("Density", fontsize=10)
    ax5.set_title("Confidence Distribution", fontsize=12)
    ax5.set_xlim(0, 1)
    
    plt.tight_layout()
    return fig

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(image).unsqueeze(0).to(device)

def get_gemini_advice(disease_name, confidence):
    """Get advisory information from Gemini about the detected disease"""
    try:
        prompt = f"""
        You are an expert plant pathologist. Provide detailed information about {disease_name} including:
        1. A brief description of the disease (50 words)
        2. Common symptoms (bullet points)
        3. Recommended treatment options (bullet points)
        4. Prevention strategies (bullet points)
        5. Any safety precautions for handling infected plants
        
        The model detected this with {confidence:.2f}% confidence. 
        If confidence is below 70%, mention that verification by an expert might be needed.
        
        Format your response in clear markdown sections.
        """
        
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error getting Gemini advice: {str(e)}")
        return "Could not retrieve advisory information at this time."

class VideoProcessor(VideoTransformerBase):
    def __init__(self, model):
        self.model = model
        self.target_layer = model.features[-2]
        self.grad_cam = GradCAM(model, self.target_layer)
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.last_update_time = time.time()
        self.update_interval = 2  # seconds between updates
        self.last_prediction = None
        self.last_heatmap = None
        
    def transform(self, frame):
        current_time = time.time()
        
        # Only process if enough time has passed since last update
        if current_time - self.last_update_time < self.update_interval:
            return frame
        
        self.last_update_time = current_time
        
        try:
            # Convert frame to PIL Image
            img = Image.fromarray(frame.to_ndarray(format="rgb24"))
            
            # Preprocess for model
            input_tensor = self.transform(img).unsqueeze(0).to(device)
            
            # Get prediction
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                top_prob, top_class = torch.max(probabilities, 1)
            
            # Generate Grad-CAM heatmap
            heatmap = self.grad_cam(input_tensor, top_class.item())
            
            # Store results
            self.last_prediction = {
                'class': CLASS_NAMES[top_class.item()],
                'confidence': top_prob.item(),
                'timestamp': current_time
            }
            self.last_heatmap = heatmap
            
            # Convert heatmap to RGB for overlay
            heatmap_img = Image.fromarray((heatmap * 255).astype('uint8')).resize(img.size)
            heatmap_rgb = np.array(heatmap_img.convert('RGB'))
            
            # Create overlay (50% opacity)
            overlay = (0.5 * frame.to_ndarray(format="rgb24") + 0.5 * heatmap_rgb).astype('uint8')
            
            return overlay
            
        except Exception as e:
            st.error(f"Error processing frame: {str(e)}")
            return frame

def display_performance_metrics(processing_time, model_size, confidence):
    """Display performance metrics in a structured way"""
    metrics = {
        "Processing Time": f"{processing_time:.2f} ms",
        "Model Size": f"{model_size:.2f} MB",
        "Prediction Confidence": f"{confidence:.2f}%",
        "Device": "GPU" if torch.cuda.is_available() else "CPU"
    }
    
    st.markdown("### ⚙️ Performance Metrics")
    cols = st.columns(4)
    for i, (name, value) in enumerate(metrics.items()):
        cols[i % 4].metric(label=name, value=value)
    
    # Additional visualizations
    with st.expander("Detailed Performance Analysis"):
        # Only plot numeric metrics
        numeric_metrics = {k: v for k, v in metrics.items() if any(char.isdigit() for char in v)}
        x = list(numeric_metrics.keys())
        y = []
        for v in numeric_metrics.values():
            try:
                y.append(float(v.split()[0]))
            except Exception:
                y.append(0)
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.bar(x, y)
        ax.set_title("System Performance")
        ax.set_ylabel("Value")
        plt.xticks(rotation=45)
        st.pyplot(fig)

def main():
    st.set_page_config(
        layout="wide", 
        page_title="LeafXplain-Lite", 
        page_icon="🌿",
        initial_sidebar_state="expanded"
    )
    
    # custom CSS for dark mode compatibility
    st.markdown("""
    <style>
    /* Main content - works in both light and dark modes */
    .main {padding: 2rem;}
    .block-container {padding-top: 2rem;}
    
    /* Cards - dark mode compatible */
    .card {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        background-color: var(--background-color);
        border: 1px solid var(--border-color);
    }
    
    /* Text colors that work in both modes */
    .card h1, .card h2, .card h3, .card p {
        color: var(--text-color) !important;
    }
    
    /* Progress bars */
    .stProgress > div > div > div {
        background-color: #4caf50;
    }
    
    /* File uploader */
    .stFileUploader > div {
        border: 2px dashed #81c784;
        border-radius: 10px;
        padding: 2rem 1rem;
        background-color: transparent;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 20px;
        background-color: var(--background-color);
        border-radius: 10px 10px 0 0;
        border: 1px solid var(--border-color);
        color: black;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #4caf50;
        color: white !important;
    }
    
    /* Theme variables */
    :root {
        --background-color: white;
        --text-color: black;
        --border-color: #e0e0e0;
    }
    
    [data-theme="dark"] {
        --background-color: #1e1e1e;
        --text-color: white;
        --border-color: #444;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .main {padding: 1rem;}
        .sidebar .sidebar-content {padding: 1rem;}
    }
    </style>
    """, unsafe_allow_html=True)
    
    # app header
    st.markdown("""
    <div class="card">
        <h1 style="margin-bottom: 0.5rem; color: var(--text-color) !important;">🌿 LeafXplain-Lite</h1>
        <p style="color: var(--text-color); margin-bottom: 0;">Advanced light-weight plant disease detection with explainability, real-time analysis and expert advisory</p>
    </div>
    """, unsafe_allow_html=True)
    
    # sidebar - Model upload
    with st.sidebar:
        st.markdown("""
        <div class="card">
            <h3>1. Upload Model</h3>
            <p style="color: var(--text-color);">Please upload your trained model file (.pth)</p>
        </div>
        """, unsafe_allow_html=True)
        
        model_file = st.file_uploader(
            "Choose a model file", 
            type="pth",
            label_visibility="collapsed",
            key="model_uploader"
        )
        
        st.markdown("""
        <div class="card">
            <h3>Detection Mode</h3>
        </div>
        """, unsafe_allow_html=True)
        
        detection_mode = st.radio(
            "Select input method:",
            ["Upload Image", "Live Camera"],
            label_visibility="collapsed"
        )
        
        st.markdown("""
        <div class="card">
            <h3>About</h3>
            <p style="color: var(--text-color);">This advanced app provides:</p>
            <ul style="color: var(--text-color);">
                <li>Plant disease classification</li>
                <li>Real-time camera detection</li>
                <li>Model attention visualization</li>
                <li>Expert advisory via Gemini</li>
                <li>Detailed performance metrics</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # main content
    if model_file is not None:
        # save uploaded model to temporary file
        model_path = "temp_model.pth"
        with open(model_path, "wb") as f:
            f.write(model_file.getbuffer())
        
        model = load_model(model_path)
        
        if model is not None:
            st.success("Model loaded successfully!", icon="✅")
            
            # Calculate model size
            model_size = os.path.getsize(model_path) / (1024 * 1024)  # in MB
            
            # Initialize Grad-CAM
            target_layer = model.features[-2]  # Layer before attention
            grad_cam = GradCAM(model, target_layer)
            
            if detection_mode == "Upload Image":
                # image upload section
                st.markdown("""
                <div class="card">
                    <h2>2. Upload Plant Image</h2>
                    <p style="color: var(--text-color);">Upload an image of a plant leaf for disease diagnosis</p>
                </div>
                """, unsafe_allow_html=True)
                
                image_file = st.file_uploader(
                    "Choose an image...", 
                    type=["jpg", "jpeg", "png"],
                    label_visibility="collapsed",
                    key="image_uploader"
                )
                
                if image_file is not None:
                    start_time = time.time()
                    image = Image.open(image_file).convert('RGB')
                    
                    # process image
                    input_tensor = preprocess_image(image)
                    
                    with torch.no_grad():
                        output = model(input_tensor)
                        probabilities = torch.softmax(output, dim=1)
                        top_prob, top_class = torch.max(probabilities, 1)
                    
                    # generate Grad-CAM heatmap
                    heatmap = grad_cam(input_tensor, top_class.item())
                    
                    processing_time = (time.time() - start_time) * 1000  # in ms
                    
                    # visualization section
                    st.markdown("""
                    <div class="card">
                        <h2>Comprehensive Analysis</h2>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Use tabs for different views
                    tab1, tab2, tab3 = st.tabs(["Diagnosis Overview", "Detailed Analysis", "Expert Advisory"])
                    
                    with tab1:
                        # combined visualization
                        fig = create_combined_figure(image, heatmap)
                        st.pyplot(fig, use_container_width=True)
                        
                        # results section
                        col1, col2 = st.columns([1, 2], gap="large")
                        
                        with col1:
                            st.markdown(f"""
                            <div class="card">
                                <h3>🔍 Diagnosis Results</h3>
                                <p style="font-size: 1.1rem; margin-bottom: 0.5rem; color: var(--text-color);"><b>Predicted Disease:</b></p>
                                <p style="font-size: 1.3rem; color: #4caf50; font-weight: bold; margin-top: 0;">{CLASS_NAMES[top_class.item()]}</p>
                                <p style="font-size: 1.1rem; margin-bottom: 0.5rem; color: var(--text-color);"><b>Confidence:</b></p>
                                <p style="font-size: 1.3rem; color: #4caf50; font-weight: bold; margin-top: 0;">{top_prob.item()*100:.2f}%</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown("""
                            <div class="card">
                                <h3>📊 Top Predictions</h3>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            probs = probabilities.squeeze().cpu().numpy()
                            top_indices = np.argsort(probs)[-5:][::-1]
                            
                            for i in top_indices:
                                label = CLASS_NAMES[i]
                                percent = probs[i]*100
                                st.markdown(f"<span style='color: var(--text-color);'><b>{label}</b></span>", unsafe_allow_html=True)
                                st.progress(float(probs[i]), text=f"{percent:.2f}%")
                        
                        # Performance metrics
                        display_performance_metrics(processing_time, model_size, top_prob.item()*100)
                    
                    with tab2:
                        # Detailed analysis with multiple visualizations
                        st.markdown("""
                        <div class="card">
                            <h3>📈 Detailed Analysis</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        detailed_fig = create_detailed_analysis_figure(image, heatmap, probabilities)
                        st.pyplot(detailed_fig, use_container_width=True)
                    
                    with tab3:
                        # Gemini advisory section
                        st.markdown("""
                        <div class="card">
                            <h3>🛡️ Expert Advisory</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with st.spinner("Consulting with plant pathology expert..."):
                            advice = get_gemini_advice(CLASS_NAMES[top_class.item()], top_prob.item()*100)
                        
                        st.markdown(advice, unsafe_allow_html=True)
            
            else:  # Live Camera mode
                st.markdown("""
                <div class="card">
                    <h2>🌿 Real-time Plant Disease Detection</h2>
                    <p style="color: var(--text-color);">Use your camera for live plant disease analysis</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.warning("Live detection may have slightly reduced accuracy compared to static image analysis.", icon="⚠️")
                
                # Create two columns - one for camera, one for results
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    ctx = webrtc_streamer(
                        key="example",
                        mode=WebRtcMode.SENDRECV,
                        video_transformer_factory=lambda: VideoProcessor(model),
                        async_transform=True,
                        media_stream_constraints={
                            "video": True,
                            "audio": False
                        }
                    )
                
                with col2:
                    if ctx.video_transformer:
                        if hasattr(ctx.video_transformer, 'last_prediction'):
                            prediction = ctx.video_transformer.last_prediction
                            
                            st.markdown("""
                            <div class="card">
                                <h3>🔍 Live Results</h3>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown(f"""
                            <p style="color: var(--text-color);"><b>Detected Disease:</b></p>
                            <p style="font-size: 1.2rem; color: #4caf50; font-weight: bold;">{prediction['class']}</p>
                            <p style="color: var(--text-color);"><b>Confidence:</b></p>
                            <p style="font-size: 1.2rem; color: #4caf50; font-weight: bold;">{prediction['confidence']*100:.2f}%</p>
                            <p style="color: var(--text-color);"><small>Last update: {time.strftime('%H:%M:%S', time.localtime(prediction['timestamp']))}</small></p>
                            """, unsafe_allow_html=True)
                            
                            # Show advisory button
                            if st.button("Get Expert Advice on This Detection"):
                                with st.spinner("Consulting with plant pathology expert..."):
                                    advice = get_gemini_advice(prediction['class'], prediction['confidence']*100)
                                st.markdown(advice, unsafe_allow_html=True)
                        
                        # Performance metrics for live mode
                        display_performance_metrics(0, model_size, ctx.video_transformer.last_prediction['confidence']*100 if hasattr(ctx.video_transformer, 'last_prediction') else 0)
        
        # clean up temporary model file
        if os.path.exists(model_path):
            os.remove(model_path)
    else:
        st.info("Please upload a trained model (.pth file) to begin diagnosis", icon="ℹ️")

if __name__ == "__main__":
    main()