import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import warnings
import google.generativeai as genai
import datetime
import json
from io import BytesIO
import base64
import pandas as pd
from geopy.geocoders import Nominatim
import time

warnings.filterwarnings('ignore')

# Configuration
CONFIG = {
    "max_history": 10,  # Number of past diagnoses to remember
    "default_language": "English",
    "languages": {
        "English": {
            "app_title": "🌿 Plant Disease Diagnosis",
            "upload_model": "Upload Model",
            "upload_image": "Upload Plant Image",
            "diagnosis_results": "Diagnosis Results",
            "top_predictions": "Top Predictions",
            "model_attention": "Model Attention",
            "original_image": "Original Image",
            "confidence": "Confidence",
            "predicted_disease": "Predicted Disease",
            "understanding_results": "Understanding the Results",
            "most_important": "Most important",
            "important": "Important",
            "less_important": "Less important",
            "remediation_advice": "Remediation Advice",
            "detailed_info": "Detailed Information",
            "history": "Diagnosis History",
            "export_report": "Export Report",
            "location": "Location (Optional)",
            "get_location": "Get Current Location",
            "enter_location": "Or enter location manually:",
            "no_model": "Please upload a trained model (.pth file) to begin diagnosis",
            "model_loaded": "Model loaded successfully!",
            "no_image": "Please upload an image for diagnosis",
            "ask_gemini": "Ask Gemini for more information",
            "gemini_question": "What would you like to know about this disease?",
            "gemini_response": "Gemini Response",
            "export_pdf": "Export as PDF",
            "export_csv": "Export as CSV",
            "capture_date": "Capture Date",
            "capture_date_placeholder": "Select date when photo was taken"
        },
        "Spanish": {
            "app_title": "🌿 Diagnóstico de Enfermedades en Plantas",
            "upload_model": "Subir Modelo",
            "upload_image": "Subir Imagen de Planta",
            "diagnosis_results": "Resultados del Diagnóstico",
            "top_predictions": "Predicciones Principales",
            "model_attention": "Atención del Modelo",
            "original_image": "Imagen Original",
            "confidence": "Confianza",
            "predicted_disease": "Enfermedad Predicha",
            "understanding_results": "Entendiendo los Resultados",
            "most_important": "Más importante",
            "important": "Importante",
            "less_important": "Menos importante",
            "remediation_advice": "Consejos de Remediación",
            "detailed_info": "Información Detallada",
            "history": "Historial de Diagnósticos",
            "export_report": "Exportar Reporte",
            "location": "Ubicación (Opcional)",
            "get_location": "Obtener Ubicación Actual",
            "enter_location": "O ingresar ubicación manualmente:",
            "no_model": "Por favor suba un modelo entrenado (.pth) para comenzar el diagnóstico",
            "model_loaded": "¡Modelo cargado exitosamente!",
            "no_image": "Por favor suba una imagen para diagnóstico",
            "ask_gemini": "Preguntar a Gemini por más información",
            "gemini_question": "¿Qué le gustaría saber sobre esta enfermedad?",
            "gemini_response": "Respuesta de Gemini",
            "export_pdf": "Exportar como PDF",
            "export_csv": "Exportar como CSV",
            "capture_date": "Fecha de Captura",
            "capture_date_placeholder": "Seleccione fecha cuando se tomó la foto"
        }
    }
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize Gemini API (replace with your actual API key)
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "your-api-key-here")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-pro')

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

# class names
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

# Remediation advice for each disease
REMEDIATION_ADVICE = {
    'Apple_Apple Scab': {
        'short': "Apply fungicides in early spring and remove fallen leaves.",
        'long': "Apple scab can be controlled through a combination of cultural practices and fungicide applications. Remove and destroy fallen leaves in autumn to reduce overwintering spores. Apply fungicides starting at green tip stage and continue through petal fall. Resistant varieties are available."
    },
    'Apple_Black Rot': {
        'short': "Prune infected branches and apply fungicides during bloom.",
        'long': "For black rot management, prune out all dead wood and cankers during dormancy. Remove mummified fruit from trees and ground. Apply fungicides during bloom period. Ensure proper tree spacing for good air circulation."
    },
    'Apple_Cedar Apple Rust': {
        'short': "Remove nearby junipers or apply fungicides in early spring.",
        'long': "Cedar apple rust requires both apple and juniper hosts. If possible, remove junipers within a 2-mile radius. If not, apply fungicides to apples starting at pink bud stage and continue through 2nd cover spray. Resistant varieties are available."
    },
    'Bell Pepper_Bacterial Spot': {
        'short': "Use disease-free seed, rotate crops, and apply copper sprays.",
        'long': "Bacterial spot is difficult to control once established. Use pathogen-free seed and transplants. Practice 2-3 year crop rotations. Avoid overhead irrigation. Copper sprays may help but are often ineffective in wet weather. Some resistant varieties exist."
    },
    'Cherry_Powdery Mildew': {
        'short': "Apply sulfur or potassium bicarbonate fungicides.",
        'long': "For powdery mildew, ensure good air circulation through proper pruning. Apply sulfur or potassium bicarbonate fungicides at first sign of disease. Avoid excessive nitrogen fertilization which promotes susceptible growth."
    },
    'Corn (Maize)_Cercospora Leaf Spot': {
        'short': "Rotate crops and use resistant hybrids.",
        'long': "Manage Cercospora leaf spot through crop rotation (at least 1 year between corn crops), tillage to bury residue, and use of resistant hybrids. Fungicides may be economical in seed production fields."
    },
    'Corn (Maize)_Common Rust': {
        'short': "Plant early and use resistant hybrids.",
        'long': "Common rust is typically not economically damaging enough to warrant fungicide application. Plant early to avoid peak disease periods. Many hybrids have good resistance. Remove volunteer corn plants that may harbor the disease."
    },
    'Corn (Maize)_Northern Leaf Blight': {
        'short': "Rotate crops, till residue, and use resistant hybrids.",
        'long': "Northern leaf blight management includes crop rotation (at least 1 year between corn crops), tillage to bury residue, and use of resistant hybrids. Fungicides may be justified when disease appears early and weather favors spread."
    },
    'Grape_Black Rot': {
        'short': "Apply fungicides from pre-bloom through 3-4 weeks after bloom.",
        'long': "Black rot control requires fungicide applications from pre-bloom through 3-4 weeks after bloom. Remove all mummies from vines and ground during pruning. Improve air circulation through proper pruning and canopy management."
    },
    'Grape_Esca (Black Measles)': {
        'short': "No cure; remove infected vines and protect pruning wounds.",
        'long': "Esca is a complex disease with no effective cure once established. Remove severely infected vines. Protect pruning wounds with fungicides to prevent infection. Avoid stress to vines through proper irrigation and nutrition."
    },
    'Grape_Leaf Blight': {
        'short': "Apply fungicides and remove infected leaves.",
        'long': "For grape leaf blight, apply fungicides at first sign of disease. Remove severely infected leaves if practical. Ensure good air circulation through proper vine spacing and canopy management. Avoid overhead irrigation."
    },
    'Peach_Bacterial Spot': {
        'short': "Apply copper sprays during dormancy and at petal fall.",
        'long': "Bacterial spot management includes copper sprays during dormancy and at petal fall. Avoid overhead irrigation. Select less susceptible varieties when possible. Maintain tree vigor through proper nutrition and irrigation but avoid excessive nitrogen."
    },
    'Potato_Early Blight': {
        'short': "Apply fungicides, rotate crops, and maintain plant health.",
        'long': "Early blight is managed through fungicide applications, 3-year crop rotations, and maintaining plant vigor through proper nutrition and irrigation. Remove volunteer potato plants and cull piles. Avoid overhead irrigation when possible."
    },
    'Potato_Late Blight': {
        'short': "Destroy infected plants and apply fungicides preventatively.",
        'long': "Late blight requires aggressive management. Destroy infected plants immediately. Apply fungicides preventatively when weather favors disease. Use certified disease-free seed potatoes. Allow tubers to mature before harvest to prevent infection."
    },
    'Strawberry_Leaf Scorch': {
        'short': "Apply fungicides and remove infected leaves after harvest.",
        'long': "For leaf scorch, apply fungicides starting at first bloom. Remove infected leaves after harvest. Ensure good air circulation through proper plant spacing. Irrigate in morning to allow leaves to dry quickly."
    },
    'Tomato_Bacterial Spot': {
        'short': "Use disease-free seed, rotate crops, and apply copper sprays.",
        'long': "Bacterial spot control requires disease-free seed and transplants. Practice 2-3 year crop rotations. Copper sprays may help but are often ineffective in wet weather. Avoid working with plants when wet. Some resistant varieties exist."
    },
    'Tomato_Early Blight': {
        'short': "Apply fungicides, stake plants, and remove infected leaves.",
        'long': "Early blight management includes regular fungicide applications, staking plants to improve air circulation, and removing infected lower leaves. Mulch to prevent soil splash. Rotate crops (3 years between tomato crops)."
    },
    'Tomato_Late Blight': {
        'short': "Destroy infected plants and apply fungicides preventatively.",
        'long': "Late blight requires immediate action. Destroy infected plants. Apply fungicides preventatively when weather favors disease. Avoid overhead irrigation. Remove volunteer tomato and potato plants. Use resistant varieties when available."
    },
    'Tomato_Septoria Leaf Spot': {
        'short': "Apply fungicides, remove infected leaves, and rotate crops.",
        'long': "Septoria leaf spot is managed through fungicide applications, removing infected lower leaves, and 3-year crop rotations. Mulch to prevent soil splash. Avoid overhead irrigation. Space plants for good air circulation."
    },
    'Tomato_Yellow Leaf Curl Virus': {
        'short': "Control whiteflies and remove infected plants.",
        'long': "Yellow leaf curl virus is spread by whiteflies. Use insecticides to control whitefly populations. Remove infected plants immediately. Use virus-free transplants. Reflective mulches may repel whiteflies. Resistant varieties are available."
    }
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
    ax1.set_title(t("original_image"), fontsize=12, pad=10, fontweight='bold', 
                 color='white' if st.get_option("theme.base") == "dark" else 'black')
    ax1.axis('off')
    
    # display original image with heatmap overlay
    ax2.imshow(img_array)
    heatmap_display = ax2.imshow(heatmap_normalized, cmap='inferno', alpha=alpha)
    ax2.set_title(t("model_attention"), fontsize=12, pad=10, fontweight='bold',
                 color='white' if st.get_option("theme.base") == "dark" else 'black')
    ax2.axis('off')
    
    # add colorbar with smaller size
    cbar = fig.colorbar(heatmap_display, ax=ax2, fraction=0.046, pad=0.01)
    cbar.ax.tick_params(labelsize=8, colors='white' if st.get_option("theme.base") == "dark" else 'black')
    
    plt.tight_layout()
    return fig

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(image).unsqueeze(0).to(device)

def t(key):
    """Translation function to get text in current language"""
    return CONFIG["languages"][st.session_state.get("language", CONFIG["default_language"])][key]

def get_gemini_response(prompt, disease):
    """Get response from Gemini API"""
    try:
        full_prompt = f"You are an agricultural expert. Provide detailed but concise information about {disease} in response to: {prompt}. Include prevention and treatment methods if relevant."
        response = gemini_model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"Could not get response from Gemini: {str(e)}"

def get_location_name(lat, lon):
    """Convert coordinates to location name"""
    try:
        geolocator = Nominatim(user_agent="plant_disease_app")
        location = geolocator.reverse(f"{lat}, {lon}")
        return location.address if location else "Unknown location"
    except:
        return "Unknown location"

def export_as_pdf(diagnosis_data, image):
    """Generate a PDF report"""
    from fpdf import FPDF
    from PIL import Image
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Title
    pdf.cell(200, 10, txt="Plant Disease Diagnosis Report", ln=1, align="C")
    pdf.ln(10)
    
    # Diagnosis information
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, txt=f"Date: {diagnosis_data['date']}", ln=1)
    if 'location' in diagnosis_data:
        pdf.cell(200, 10, txt=f"Location: {diagnosis_data['location']}", ln=1)
    pdf.cell(200, 10, txt=f"Predicted Disease: {diagnosis_data['disease']}", ln=1)
    pdf.cell(200, 10, txt=f"Confidence: {diagnosis_data['confidence']:.2f}%", ln=1)
    pdf.ln(10)
    
    # Remediation advice
    pdf.set_font("Arial", 'B', size=10)
    pdf.cell(200, 10, txt="Remediation Advice:", ln=1)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 10, txt=REMEDIATION_ADVICE.get(diagnosis_data['disease'], {}).get('long', 'No specific advice available'))
    
    # Save image to temporary file
    img_path = "temp_image.jpg"
    image.save(img_path)
    
    # Add image to PDF
    pdf.ln(10)
    pdf.image(img_path, x=50, w=100)
    
    # Remove temporary image
    if os.path.exists(img_path):
        os.remove(img_path)
    
    return pdf.output(dest='S').encode('latin1')

def init_session_state():
    """Initialize session state variables"""
    if 'diagnosis_history' not in st.session_state:
        st.session_state.diagnosis_history = []
    if 'language' not in st.session_state:
        st.session_state.language = CONFIG["default_language"]
    if 'gemini_questions' not in st.session_state:
        st.session_state.gemini_questions = {}

def main():
    st.set_page_config(
        layout="wide", 
        page_title="Plant Disease Diagnosis", 
        page_icon="🌿",
        initial_sidebar_state="expanded"
    )
    
    init_session_state()
    
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
    
    /* History items */
    .history-item {
        border-bottom: 1px solid var(--border-color);
        padding: 0.5rem 0;
    }
    .history-item:last-child {
        border-bottom: none;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Language selector in sidebar
    with st.sidebar:
        st.session_state.language = st.selectbox(
            "Language",
            options=list(CONFIG["languages"].keys()),
            index=list(CONFIG["languages"].keys()).index(st.session_state.language)
        )
    
    # app header
    st.markdown(f"""
    <div class="card">
        <h1 style="margin-bottom: 0.5rem; color: var(--text-color) !important;">{t("app_title")}</h1>
        <p style="color: var(--text-color); margin-bottom: 0;">Upload a plant image to detect diseases and visualize model attention</p>
    </div>
    """, unsafe_allow_html=True)
    
    # sidebar - Model upload and history
    with st.sidebar:
        st.markdown(f"""
        <div class="card">
            <h3>1. {t("upload_model")}</h3>
            <p style="color: var(--text-color);">Please upload your trained model file (.pth)</p>
        </div>
        """, unsafe_allow_html=True)
        
        model_file = st.file_uploader(
            "Choose a model file", 
            type="pth",
            label_visibility="collapsed"
        )
        
        # Diagnosis history
        if st.session_state.diagnosis_history:
            st.markdown(f"""
            <div class="card">
                <h3>{t("history")}</h3>
                <div style="max-height: 300px; overflow-y: auto;">
            """, unsafe_allow_html=True)
            
            for i, item in enumerate(reversed(st.session_state.diagnosis_history[-CONFIG["max_history"]:])):
                with st.expander(f"{item['date']} - {item['disease']} ({item['confidence']:.1f}%)", expanded=False):
                    st.image(item['image'], use_column_width=True)
                    st.write(f"**{t('confidence')}:** {item['confidence']:.1f}%")
                    if 'location' in item:
                        st.write(f"**{t('location')}:** {item['location']}")
                    if st.button(f"View details {i}", key=f"history_{i}"):
                        st.session_state.current_diagnosis = item
            
            st.markdown("</div></div>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card">
            <h3>About</h3>
            <p style="color: var(--text-color);">This app uses deep learning to:</p>
            <ul style="color: var(--text-color);">
                <li>Classify plant diseases</li>
                <li>Visualize model attention</li>
                <li>Explain predictions</li>
                <li>Provide remediation advice</li>
                <li>Offer expert consultation</li>
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
            st.success(t("model_loaded"), icon="✅")
            
            # initialize Grad-CAM
            target_layer = model.features[-2]  # Layer before attention
            grad_cam = GradCAM(model, target_layer)
            
            # image upload section
            st.markdown(f"""
            <div class="card">
                <h2>2. {t("upload_image")}</h2>
                <p style="color: var(--text-color);">Upload an image of a plant leaf for disease diagnosis</p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                image_file = st.file_uploader(
                    "Choose an image...", 
                    type=["jpg", "jpeg", "png"],
                    label_visibility="collapsed"
                )
            
            with col2:
                # Optional location information
                location = None
                if st.checkbox(t("location")):
                    if st.button(t("get_location")):
                        try:
                            # This would require browser geolocation permissions
                            # In a real app, you'd use JavaScript to get this
                            # For demo purposes, we'll simulate it
                            lat, lon = 37.7749, -122.4194  # San Francisco coordinates
                            location = get_location_name(lat, lon)
                            st.success(f"Location detected: {location}")
                        except:
                            st.warning("Could not get current location")
                    
                    st.write(t("enter_location"))
                    manual_location = st.text_input("", label_visibility="collapsed")
                    if manual_location:
                        location = manual_location
                
                # Optional capture date
                capture_date = st.date_input(t("capture_date"), 
                                          value=datetime.date.today(),
                                          help=t("capture_date_placeholder"))
            
            if image_file is not None:
                image = Image.open(image_file).convert('RGB')
                
                # process image
                input_tensor = preprocess_image(image)
                
                with torch.no_grad():
                    output = model(input_tensor)
                    probabilities = torch.softmax(output, dim=1)
                    top_prob, top_class = torch.max(probabilities, 1)
                
                # generate Grad-CAM heatmap
                heatmap = grad_cam(input_tensor, top_class.item())
                
                # Store current diagnosis
                current_diagnosis = {
                    'date': capture_date.strftime("%Y-%m-%d %H:%M"),
                    'disease': CLASS_NAMES[top_class.item()],
                    'confidence': top_prob.item()*100,
                    'image': image,
                    'probabilities': probabilities.squeeze().cpu().numpy().tolist()
                }
                
                if location:
                    current_diagnosis['location'] = location
                
                # Add to history if not already there
                if not any(d['date'] == current_diagnosis['date'] and 
                          d['disease'] == current_diagnosis['disease'] for d in st.session_state.diagnosis_history):
                    st.session_state.diagnosis_history.append(current_diagnosis)
                
                # visualization section
                st.markdown(f"""
                <div class="card">
                    <h2>{t("image_analysis")}</h2>
                </div>
                """, unsafe_allow_html=True)
                
                # combined visualization
                fig = create_combined_figure(image, heatmap)
                st.pyplot(fig, use_container_width=True)
                
                # results section
                col1, col2 = st.columns([1, 2], gap="large")
                
                with col1:
                    st.markdown(f"""
                    <div class="card">
                        <h3>🔍 {t("diagnosis_results")}</h3>
                        <p style="font-size: 1.1rem; margin-bottom: 0.5rem; color: var(--text-color);"><b>{t("predicted_disease")}:</b></p>
                        <p style="font-size: 1.3rem; color: #4caf50; font-weight: bold; margin-top: 0;">{CLASS_NAMES[top_class.item()]}</p>
                        <p style="font-size: 1.1rem; margin-bottom: 0.5rem; color: var(--text-color);"><b>{t("confidence")}:</b></p>
                        <p style="font-size: 1.3rem; color: #4caf50; font-weight: bold; margin-top: 0;">{top_prob.item()*100:.2f}%</p>
                    </div>
                    
                    <div class="card">
                        <h3>💡 {t("remediation_advice")}</h3>
                        <p style="color: var(--text-color);">{REMEDIATION_ADVICE.get(CLASS_NAMES[top_class.item()], {}).get('short', 'No specific advice available')}</p>
                        <div style="margin-top: 1rem;">
                            <button onclick="window.scrollTo(0, document.body.scrollHeight);" style="background-color: #4caf50; color: white; border: none; padding: 0.5rem 1rem; border-radius: 4px; cursor: pointer;">See detailed advice</button>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="card">
                        <h3>📊 {t("top_predictions")}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    probs = probabilities.squeeze().cpu().numpy()
                    top_indices = np.argsort(probs)[-5:][::-1]
                    
                    for i in top_indices:
                        label = CLASS_NAMES[i]
                        percent = probs[i]*100
                        st.markdown(f"<span style='color: var(--text-color);'><b>{label}</b></span>", unsafe_allow_html=True)
                        st.progress(float(probs[i]), text=f"{percent:.2f}%")
                
                # interpretation section
                st.markdown(f"""
                <div class="card">
                    <h3>🔎 {t("understanding_results")}</h3>
                    <p style="color: var(--text-color);">The attention map shows which areas of the image most influenced the model's prediction:</p>
                    <div style="display: flex; justify-content: center; margin: 1rem 0;">
                        <div style="text-align: center; margin: 0 1rem;">
                            <div style="width: 20px; height: 20px; background-color: #d62728; display: inline-block; border-radius: 3px;"></div>
                            <p style="margin: 0.2rem 0; font-size: 0.9rem; color: var(--text-color);">{t("most_important")}</p>
                        </div>
                        <div style="text-align: center; margin: 0 1rem;">
                            <div style="width: 20px; height: 20px; background-color: #ff7f0e; display: inline-block; border-radius: 3px;"></div>
                            <p style="margin: 0.2rem 0; font-size: 0.9rem; color: var(--text-color);">{t("important")}</p>
                        </div>
                        <div style="text-align: center; margin: 0 1rem;">
                            <div style="width: 20px; height: 20px; background-color: #1f77b4; display: inline-block; border-radius: 3px;"></div>
                            <p style="margin: 0.2rem 0; font-size: 0.9rem; color: var(--text-color);">{t("less_important")}</p>
                        </div>
                    </div>
                    <p style="color: var(--text-color);">The color intensity corresponds to the relative importance of each region in the diagnosis.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Detailed information and Gemini integration
                st.markdown(f"""
                <div class="card">
                    <h3>📚 {t("detailed_info")}</h3>
                    <p style="color: var(--text-color);">{REMEDIATION_ADVICE.get(CLASS_NAMES[top_class.item()], {}).get('long', 'No detailed information available.')}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Gemini API integration
                if GEMINI_API_KEY != "your-api-key-here":
                    st.markdown(f"""
                    <div class="card">
                        <h3>🤖 {t("ask_gemini")}</h3>
                        <p style="color: var(--text-color);">{t("gemini_question")}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    question = st.text_input("", label_visibility="collapsed",
                                            placeholder="Type your question here...")
                    
                    if question:
                        # Store question in session state for this disease
                        disease = CLASS_NAMES[top_class.item()]
                        if disease not in st.session_state.gemini_questions:
                            st.session_state.gemini_questions[disease] = []
                        st.session_state.gemini_questions[disease].append(question)
                        
                        with st.spinner("Getting response from Gemini..."):
                            gemini_response = get_gemini_response(question, disease)
                        st.markdown(f"""
                        <div class="card">
                            <h3>💬 {t("gemini_response")}</h3>
                            <p style="color: var(--text-color);">{gemini_response}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Export options
                st.markdown(f"""
                <div class="card">
                    <h3>📤 {t("export_report")}</h3>
                </div>
                """, unsafe_allow_html=True)
                col_pdf, col_csv = st.columns(2)
                with col_pdf:
                    if st.button(t("export_pdf")):
                        pdf_bytes = export_as_pdf(current_diagnosis, image)
                        b64_pdf = base64.b64encode(pdf_bytes).decode()
                        href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="diagnosis_report.pdf">Download PDF Report</a>'
                        st.markdown(href, unsafe_allow_html=True)
                with col_csv:
                    if st.button(t("export_csv")):
                        # Prepare CSV data
                        csv_data = {
                            "Date": [current_diagnosis['date']],
                            "Disease": [current_diagnosis['disease']],
                            "Confidence": [current_diagnosis['confidence']],
                            "Location": [current_diagnosis.get('location', '')]
                        }
                        df = pd.DataFrame(csv_data)
                        csv_bytes = df.to_csv(index=False).encode()
                        b64_csv = base64.b64encode(csv_bytes).decode()
                        href = f'<a href="data:file/csv;base64,{b64_csv}" download="diagnosis_report.csv">Download CSV Report</a>'
                        st.markdown(href, unsafe_allow_html=True)
        else:
            st.warning(t("no_model"))
    else:
        st.info(t("no_model"))

if __name__ == "__main__":
    main()