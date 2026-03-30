from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image, ImageFilter, ImageEnhance
import io
import uvicorn
from typing import List, Optional
import time
import os
from pathlib import Path
import cv2
import numpy as np
import base64
import requests
from urllib.parse import urlparse
from groq import Groq
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Configure Groq
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ==================== CONFIGURATION ====================
BASE_DIR = Path(__file__).parent.absolute()
MODEL_PATH = BASE_DIR / "best_model.pth"
CLASS_NAMES_PATH = BASE_DIR / "class_names.txt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 27

# ==================== INITIALIZE FASTAPI ====================
app = FastAPI(
    title="🌱 Crop Disease Detection API",
    description="ResNet50 for crop disease detection",
    version="2.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001", "http://localhost:5000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== GLOBAL VARIABLES ====================
model = None
class_names = []
device = DEVICE

# ==================== STANDARD TRANSFORMS (MATCHES TRAINING) ====================
# This is EXACTLY what your model was trained on
standard_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ==================== SMART PREPROCESSING DETECTOR ====================

class SmartPreprocessor:
    """
    Intelligently detects if image needs preprocessing:
    - If image already looks like training data (grey background, single leaf) → use standard transform
    - If image has complex background → apply leaf segmentation
    """
    
    def __init__(self, target_size=224, grey_bg_value=128):
        self.target_size = target_size
        self.grey_bg_value = grey_bg_value
        self.debug_images = {}
        
    def needs_preprocessing(self, image):
        """
        Determine if image needs preprocessing
        Returns: (bool, reason)
        """
        # Convert to numpy
        if isinstance(image, Image.Image):
            img_np = np.array(image)
        else:
            img_np = image
            
        # Handle different channels
        if len(img_np.shape) == 2:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        elif img_np.shape[2] == 4:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
        
        # 1. Check if background is already grey-like
        # Sample corners of image
        h, w = img_np.shape[:2]
        corners = [
            img_np[0:20, 0:20].mean(axis=(0,1)),  # top-left
            img_np[0:20, w-20:w].mean(axis=(0,1)),  # top-right
            img_np[h-20:h, 0:20].mean(axis=(0,1)),  # bottom-left
            img_np[h-20:h, w-20:w].mean(axis=(0,1))  # bottom-right
        ]
        
        # Calculate average corner color
        avg_corner_color = np.mean(corners, axis=0)
        
        # Check if corners are greyish (similar R,G,B values)
        color_std = np.std(avg_corner_color)
        is_grey_background = color_std < 30  # Low variation = greyish
        
        # Check if corners are close to 128 (mid-grey)
        grey_distance = np.abs(avg_corner_color - 128).mean()
        is_grey_background = is_grey_background and grey_distance < 50
        
        # 2. Check if there's a single dominant object (leaf)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Count number of distinct regions
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        significant_contours = [c for c in contours if cv2.contourArea(c) > 1000]
        
        has_single_leaf = len(significant_contours) <= 3  # Leaf + maybe small noise
        
        # 3. Check color variance (complex backgrounds have high variance)
        color_variance = np.var(img_np.reshape(-1, 3), axis=0).mean()
        has_complex_background = color_variance > 3000
        
        # Decision logic
        if is_grey_background and has_single_leaf:
            return False, "Image already matches training data (grey background, single leaf)"
        elif has_complex_background or not has_single_leaf:
            return True, f"Complex background detected (variance: {color_variance:.0f})"
        else:
            # When unsure, use standard transform first, fall back to preprocessing if confidence low
            return True, "Image may benefit from preprocessing"
    
    def __call__(self, image, force_preprocess=False, return_debug=False):
        """
        Process image with intelligent detection
        """
        self.debug_images = {}
        
        # Check if preprocessing is needed
        if not force_preprocess:
            needs_it, reason = self.needs_preprocessing(image)
            print(f"🔍 Preprocessing check: {reason}")
            
            if not needs_it:
                # Image already looks like training data, just resize/crop
                if isinstance(image, Image.Image):
                    img_np = np.array(image)
                else:
                    img_np = image
                    
                # Simple resize and crop
                pil_img = Image.fromarray(img_np) if isinstance(img_np, np.ndarray) else image
                
                # Apply standard resize/crop
                resized = transforms.Resize(256)(pil_img)
                cropped = transforms.CenterCrop(224)(resized)
                
                if return_debug:
                    self.debug_images['original'] = img_np
                    self.debug_images['final'] = np.array(cropped)
                    return cropped, self.debug_images
                return cropped
        
        # If we get here, apply full preprocessing
        return self._full_preprocess(image, return_debug)
    
    def _full_preprocess(self, image, return_debug=False):
        """
        Full preprocessing pipeline for internet images
        """
        # Convert to numpy
        if isinstance(image, Image.Image):
            img_np = np.array(image)
        else:
            img_np = image
            
        self.debug_images['original'] = img_np.copy()
        
        # Normalize channels
        img_np = self._normalize_channels(img_np)
        
        # Segment leaf
        leaf_mask = self._segment_leaf(img_np)
        self.debug_images['mask'] = leaf_mask
        
        # Place on grey background
        img_grey = self._place_on_grey_background(img_np, leaf_mask)
        self.debug_images['grey_bg'] = img_grey.copy()
        
        # Center and resize
        img_final = self._center_and_resize(img_grey, leaf_mask)
        self.debug_images['final'] = img_final.copy()
        
        result = Image.fromarray(img_final)
        
        if return_debug:
            return result, self.debug_images
        return result
    
    def _normalize_channels(self, img):
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        return img
    
    def _segment_leaf(self, img):
        """Simple but effective leaf segmentation"""
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
        # Green color range
        lower_green = np.array([30, 30, 30])
        upper_green = np.array([90, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Clean up
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Keep largest contour only
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            leaf_mask = np.zeros_like(mask)
            cv2.drawContours(leaf_mask, [largest], -1, 255, -1)
            return leaf_mask
        
        return mask
    
    def _place_on_grey_background(self, img, mask):
        grey_bg = np.ones_like(img) * self.grey_bg_value
        mask_3channel = np.stack([mask/255.0, mask/255.0, mask/255.0], axis=2)
        result = (img * mask_3channel + grey_bg * (1 - mask_3channel)).astype(np.uint8)
        return result
    
    def _center_and_resize(self, img, mask):
        y_indices, x_indices = np.where(mask > 0)
        
        if len(x_indices) == 0 or len(y_indices) == 0:
            return cv2.resize(img, (self.target_size, self.target_size))
        
        x_min, x_max = x_indices.min(), x_indices.max()
        y_min, y_max = y_indices.min(), y_indices.max()
        
        # Add padding
        pad_x = int((x_max - x_min) * 0.1)
        pad_y = int((y_max - y_min) * 0.1)
        
        x_min = max(0, x_min - pad_x)
        x_max = min(img.shape[1], x_max + pad_x)
        y_min = max(0, y_min - pad_y)
        y_max = min(img.shape[0], y_max + pad_y)
        
        # Crop and resize
        leaf_cropped = img[y_min:y_max, x_min:x_max]
        h, w = leaf_cropped.shape[:2]
        aspect = w / h
        
        if aspect > 1:
            new_w = self.target_size
            new_h = int(self.target_size / aspect)
        else:
            new_h = self.target_size
            new_w = int(self.target_size * aspect)
        
        leaf_resized = cv2.resize(leaf_cropped, (new_w, new_h))
        
        # Center on grey canvas
        canvas = np.ones((self.target_size, self.target_size, 3), dtype=np.uint8) * self.grey_bg_value
        x_offset = (self.target_size - new_w) // 2
        y_offset = (self.target_size - new_h) // 2
        
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = leaf_resized
        return canvas

# Initialize smart preprocessor
smart_preprocessor = SmartPreprocessor()
# ==================== GROQ REMEDY FUNCTION ====================
def get_remedy(disease_name: str) -> str:
    try:
        clean_name = disease_name.replace("___", " ").replace("_", " ").title()
        
        prompt = f"""
        A crop has been diagnosed with: {clean_name}

        Please provide:
        1. Brief description of this disease (1-2 sentences)
        2. Immediate remedies (both organic and chemical options)
        3. Prevention tips for the future

        Keep it concise and practical for a farmer. Use simple language.
        Use plain text only.
        """
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )
        return response.choices[0].message.content

    except Exception as e:
        print(f"❌ GROQ ERROR: {str(e)}")
        return f"Remedy unavailable. Please consult a local agricultural expert."

# ==================== MODEL ARCHITECTURE ====================
def create_model(num_classes):
    """Creates the exact same architecture as training"""
    model = resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    return model

# ==================== LOAD MODEL ====================
@app.on_event("startup")
async def startup_event():
    """Load model when server starts"""
    global model, class_names, NUM_CLASSES
    
    print("\n" + "="*70)
    print("🌱 CROP DISEASE DETECTION API - SMART PREPROCESSING")
    print("="*70)
    
    # Load class names
    if CLASS_NAMES_PATH.exists():
        with open(CLASS_NAMES_PATH, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        NUM_CLASSES = len(class_names)
        print(f"📋 Loaded {NUM_CLASSES} classes")
    else:
        # Your class list here
        class_names = [
            "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
            "Corn___Cercospora_leaf_spot", "Corn___Common_rust", "Corn___Northern_Leaf_Blight", "Corn___healthy",
            "Grape___Black_rot", "Grape___Esca", "Grape___Leaf_blight", "Grape___healthy",
            "Pepper___Bacterial_spot", "Pepper___healthy",
            "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
            "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
            "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites",
            "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
            "Tomato___healthy"
        ]
        NUM_CLASSES = len(class_names)
    
    # Load model
    try:
        print(f"\n📂 Loading model from: {MODEL_PATH}")
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            if 'val_acc' in checkpoint:
                print(f"📊 Best validation accuracy: {checkpoint['val_acc']:.2f}%")
        else:
            state_dict = checkpoint
        
        model = create_model(NUM_CLASSES)
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device)
        model.eval()
        
        print(f"✅ Model loaded successfully on {device}")
        
        # Test with potato blight image (simulated)
        dummy = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            output = model(dummy)
        print(f"✅ Test inference: {output.shape}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise
    
    print("="*70 + "\n")

# ==================== PREDICTION FUNCTION ====================
async def predict_image(image: Image.Image, use_preprocessing: bool = None):
    """
    Unified prediction function
    """
    global model
    
    # Decide whether to preprocess
    if use_preprocessing is None:
        # Auto-detect
        needs_it, reason = smart_preprocessor.needs_preprocessing(image)
        print(f"🤔 Auto-detection: {reason}")
        use_preprocessing = needs_it
    
    # Apply preprocessing if needed
    if use_preprocessing:
        print("🔄 Applying full preprocessing")
        processed_img = smart_preprocessor(image)
    else:
        print("✓ Using standard transform (no preprocessing)")
        processed_img = image
    
    # Apply standard transform
    input_tensor = standard_transform(processed_img).unsqueeze(0).to(device)
    
    # Predict
    start_time = time.time()
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    
    inference_time = (time.time() - start_time) * 1000
    
    # Get top predictions
    top_probs, top_indices = torch.topk(probabilities, 3)
    
    return {
        "probabilities": probabilities,
        "top_probs": top_probs,
        "top_indices": top_indices,
        "inference_time": inference_time,
        "preprocessing_applied": use_preprocessing
    }

# ==================== API ENDPOINTS ====================

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head><title>🌱 Crop Disease API</title></head>
        <body>
            <h1>🌱 Crop Disease Detection API</h1>
            <p>Smart preprocessing that preserves potato blight accuracy!</p>
            <ul>
                <li><a href="/docs">API Documentation</a></li>
                <li><a href="/test-potato">Test Potato Blight</a></li>
                <li><a href="/predict-page">Prediction Page</a></li>
            </ul>
        </body>
    </html>
    """

@app.get("/test-potato", response_class=HTMLResponse)
async def test_potato_page():
    """Test page specifically for potato blight"""
    return """
    <html>
    <head>
        <title>🥔 Potato Blight Test</title>
        <style>
            body { font-family: Arial; max-width: 800px; margin: 50px auto; padding: 20px; }
            .card { border: 1px solid #ddd; padding: 20px; border-radius: 8px; margin: 20px 0; }
            button { padding: 10px 20px; background: #4CAF50; color: white; border: none; cursor: pointer; }
        </style>
    </head>
    <body>
        <h1>🥔 Potato Blight Detection Test</h1>
        
        <div class="card">
            <h3>Test with Standard Processing (Should work best)</h3>
            <form action="/predict-upload" method="post" enctype="multipart/form-data">
                <input type="hidden" name="preprocessing" value="false">
                <input type="file" name="file" accept="image/*" required>
                <button type="submit">Test (No Preprocessing)</button>
            </form>
        </div>
        
        <div class="card">
            <h3>Test with Auto Detection (Recommended)</h3>
            <form action="/predict-upload" method="post" enctype="multipart/form-data">
                <input type="hidden" name="preprocessing" value="auto">
                <input type="file" name="file" accept="image/*" required>
                <button type="submit">Test (Auto)</button>
            </form>
        </div>
        
        <div class="card">
            <h3>Test with Full Preprocessing</h3>
            <form action="/predict-upload" method="post" enctype="multipart/form-data">
                <input type="hidden" name="preprocessing" value="true">
                <input type="file" name="file" accept="image/*" required>
                <button type="submit">Test (Full Preprocessing)</button>
            </form>
        </div>
        
        <p><a href="/">← Back</a></p>
    </body>
    </html>
    """

@app.post("/predict-upload")
async def predict_upload(
    file: UploadFile = File(...),
    preprocessing: str = "auto"
):
    """
    Predict with control over preprocessing
    preprocessing: "auto", "true", "false"
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Determine preprocessing
        use_preprocessing = None
        if preprocessing == "true":
            use_preprocessing = True
        elif preprocessing == "false":
            use_preprocessing = False
        
        # Predict
        result = await predict_image(image, use_preprocessing)
        
        # Format predictions
        predictions = []
        for i in range(3):
            idx = result['top_indices'][i].item()
            class_name = class_names[idx]
            
            if "___" in class_name:
                crop, disease = class_name.split("___")
                display_name = f"{crop} - {disease.replace('_', ' ')}"
            else:
                display_name = class_name.replace('_', ' ')
            
            predictions.append({
                "rank": i + 1,
                "class": class_name,
                "display_name": display_name,
                "confidence": float(result['top_probs'][i]),
                "confidence_percentage": round(float(result['top_probs'][i]) * 100, 2)
            })
        
       # Get remedy for top predicted disease
        top_disease = predictions[0]["class"]
        remedy = get_remedy(top_disease)

        return JSONResponse({
            "success": True,
            "filename": file.filename,
            "preprocessing_applied": result['preprocessing_applied'],
            "inference_time_ms": round(result['inference_time'], 2),
            "predictions": predictions,
            "diagnosed_disease": predictions[0]["display_name"],
            "remedy": remedy
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Standard prediction with auto preprocessing"""
    return await predict_upload(file, "auto")

@app.get("/predict-url")
async def predict_url(url: str):
    """Predict from URL"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Download image
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        image = Image.open(io.BytesIO(response.content)).convert('RGB')
        
        # Predict with auto preprocessing
        result = await predict_image(image, None)
        
        # Format response
        predictions = []
        for i in range(3):
            idx = result['top_indices'][i].item()
            class_name = class_names[idx]
            predictions.append({
                "rank": i + 1,
                "class": class_name,
                "confidence": float(result['top_probs'][i]),
                "confidence_percentage": round(float(result['top_probs'][i]) * 100, 2)
            })
        
       # Get remedy for top prediction
        top_disease = predictions[0]["class"]
        remedy = get_remedy(top_disease)

        return JSONResponse({
            "success": True,
            "filename": file.filename,
            "preprocessing_applied": result['preprocessing_applied'],
            "inference_time_ms": round(result['inference_time'], 2),
            "predictions": predictions,
            "remedy": remedy,                          # ← NEW
            "diagnosed_disease": predictions[0]["display_name"]  # ← NEW
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if model else "degraded",
        "device": str(device),
        "model_loaded": model is not None,
        "classes_loaded": len(class_names)
    }

@app.get("/classes")
async def get_classes():
    """Get all classes"""
    class_info = []
    for i, name in enumerate(class_names):
        class_info.append({
            "index": i,
            "class_name": name,
            "display_name": name.replace("___", " - ").replace("_", " ")
        })
    
    return {
        "count": len(class_names),
        "classes": class_info
    }

# ==================== RUN SERVER ====================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🌱 STARTING CROP DISEASE DETECTION API")
    print("="*70)
    print(f"📂 Model path: {MODEL_PATH}")
    print(f"💻 Device: {DEVICE}")
    print(f"🤖 Smart preprocessing: ON")
    print(f"🥔 Potato blight optimized: YES")
    print(f"🌐 Server: http://localhost:8000")
    print("="*70 + "\n")
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )