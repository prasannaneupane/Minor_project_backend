"""
Plant Disease Segmentation API using SegFormer
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import io
import base64
import uvicorn
from typing import Dict, Any, Optional
import time
import os
import matplotlib.pyplot as plt
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import pytorch_lightning as pl
from pathlib import Path
# ==================== CONFIGURATION ====================
# Get the current directory of this script
BASE_DIR = Path(__file__).parent.absolute()

# Model paths - Using Path for cross-platform compatibility
MODEL_PATH = BASE_DIR / "best.ckpt"  # If model is directly in backend folder
# If on Windows, use a Windows-style path or adjust accordingly
# MODEL_PATH = "D:/Plant_Disease_Detection/Split(0.7,0.2,0.1), LR (1e-5), WD(1e-3) 20 epoc NL/best.ckpt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 2  # Background (0) and Disease (1)
IMG_SIZE = 512

print(f"📦 Using device: {DEVICE}")

# ==================== INITIALIZE FASTAPI ====================
app = FastAPI(
    title="🌿 Plant Disease Segmentation API",
    description="SegFormer model for plant disease segmentation and severity analysis",
    version="1.0.0"
)

# Enable CORS for MERN frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:5000", "*"],  # Add frontend URLs here
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== GLOBAL VARIABLES ====================
model = None
processor = None

# ==================== DEFINE LOSS CLASSES (for loading) ====================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        targets_one_hot = torch.nn.functional.one_hot(
            targets, num_classes=probs.shape[1]
        ).permute(0, 3, 1, 2).float()
        intersection = (probs * targets_one_hot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

# ==================== DEFINE LIGHTNING MODEL ====================
class SegformerLightning(pl.LightningModule):
    def __init__(self, num_classes=2, lr=1e-5, freeze_epochs=3):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b1-finetuned-ade-512-512",
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        
        # Add dropout for regularization
        self.model.decode_head.dropout = nn.Dropout2d(0.3)
        
        # Loss functions
        self.ce = nn.CrossEntropyLoss(label_smoothing=0.05, ignore_index=255)
        self.dice = DiceLoss()
        
        # For inference only, we don't need metrics here
    
    def forward(self, pixel_values=None, **kwargs):
        if pixel_values is not None:
            return self.model(pixel_values=pixel_values, **kwargs)
        return self.model(**kwargs)
    
    def training_step(self, batch, batch_idx):
        # Not used for inference
        pass
    
    def configure_optimizers(self):
        # Not used for inference
        pass

# ==================== LOAD MODEL ====================
def load_model():
    """Load the SegFormer model from checkpoint"""
    global model, processor
    
    print("\n" + "="*60)
    print("🌿 LOADING SEGMENTATION MODEL")
    print("="*60)
    
    try:
        # Check if model file exists
        if not os.path.exists(MODEL_PATH):
            print(f"⚠️  Model not found at: {MODEL_PATH}")
            print("   Please update MODEL_PATH in the script")
            return False
        
        print(f"📂 Loading checkpoint from: {MODEL_PATH}")
        
        # Load processor
        processor = SegformerImageProcessor.from_pretrained(
            "nvidia/segformer-b1-finetuned-ade-512-512"
        )
        print("✅ Processor loaded")
        
        # Load model
        model = SegformerLightning.load_from_checkpoint(
            MODEL_PATH,
            map_location=DEVICE
        )
        model.eval()
        model = model.to(DEVICE)
        
        print(f"✅ Model loaded on {DEVICE}")
        
        # Test with dummy input
        dummy = torch.randn(1, 3, 512, 512).to(DEVICE)
        with torch.no_grad():
            output = model(pixel_values=dummy)
        print(f"✅ Test inference successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

# ==================== HELPER FUNCTIONS ====================
def get_leaf_mask(image: np.ndarray) -> np.ndarray:
    """Simple green thresholding to get leaf region"""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])
    leaf_mask = cv2.inRange(hsv, lower_green, upper_green)
    return leaf_mask

def calculate_severity(leaf_mask: np.ndarray, disease_mask: np.ndarray) -> float:
    """Compute severity percentage"""
    leaf_pixels = np.sum(leaf_mask > 0)
    disease_pixels = np.sum(disease_mask == 1)
    
    if leaf_pixels == 0:
        return 0.0
    
    severity = (disease_pixels / leaf_pixels) * 100
    return min(severity, 100.0)

def separate_disease_regions(disease_mask: np.ndarray, image_rgb: np.ndarray) -> Dict[str, Any]:
    """Apply watershed to separate individual disease regions"""
    
    # Morphological cleaning
    kernel = np.ones((5,5), np.uint8)
    opening = cv2.morphologyEx(disease_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Distance transform
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 1, 0)
    sure_fg = np.uint8(sure_fg)
    
    # Background and unknown regions
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Markers for watershed
    num_markers, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 1] = 0
    
    # Apply watershed
    image_for_ws = image_rgb.copy()
    markers = cv2.watershed(image_for_ws, markers)
    
    # Count disease regions
    regions = []
    for label in np.unique(markers):
        if label <= 1:
            continue
        mask = (markers == label).astype(np.uint8)
        if np.sum(mask) < 40:
            continue
        regions.append({
            "label": int(label),
            "pixel_count": int(np.sum(mask)),
            "area_percentage": float(np.sum(mask) / disease_mask.size * 100)
        })
    
    return {
        "num_regions": len(regions),
        "regions": regions,
        "markers": markers
    }

def create_visualization(image_rgb: np.ndarray, disease_mask: np.ndarray, markers: np.ndarray) -> str:
    """Create visualization with disease regions highlighted"""
    
    # Create overlay
    overlay = image_rgb.copy()
    
    for label in np.unique(markers):
        if label <= 1:
            continue
        mask = (markers == label).astype(np.uint8)
        if np.sum(mask) < 40:
            continue
        # Color each disease region red
        overlay[mask == 1] = [255, 0, 0]
    
    # Blend with original
    alpha = 0.45
    colored_output = cv2.addWeighted(overlay, alpha, image_rgb, 1 - alpha, 0)
    
    # Create side-by-side comparison
    h, w = image_rgb.shape[:2]
    comparison = np.zeros((h, w*3, 3), dtype=np.uint8)
    comparison[:, :w] = image_rgb
    comparison[:, w:w*2] = cv2.cvtColor((disease_mask * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    comparison[:, w*2:] = colored_output
    
    # Convert to base64 for sending to frontend
    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return img_base64

# ==================== STARTUP EVENT ====================
@app.on_event("startup")
async def startup_event():
    """Load model when server starts"""
    global model, processor
    
    print("\n" + "="*60)
    print("🌿 PLANT DISEASE SEGMENTATION API")
    print("="*60)
    
    success = load_model()
    
    if success:
        print(f"\n✅ API ready for predictions on port 8001!")
    else:
        print(f"\n⚠️  API running but model not loaded. Check model path.")
    
    print("="*60 + "\n")

# ==================== API ENDPOINTS ====================
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "🌿 Plant Disease Segmentation API",
        "version": "1.0.0",
        "model": "SegFormer-B1",
        "task": "Disease Segmentation & Severity Analysis",
        "device": str(DEVICE),
        "model_loaded": model is not None,
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "severity": "/severity (POST)",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if model else "degraded",
        "device": str(DEVICE),
        "model_loaded": model is not None,
        "cuda_available": torch.cuda.is_available()
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Run disease segmentation on uploaded image
    
    Returns:
    - Disease mask
    - Severity percentage
    - Number of disease regions
    - Visualization image (base64)
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check model path.")
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        print("📂 Reading uploaded file...")
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        
        print("✅ Image successfully decoded")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        start_time = time.time()
        
        # Prepare input for model
        print("📊 Preparing input for model...")
        inputs = processor(images=image_rgb, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(DEVICE)
        
        # Run inference
        print("🤖 Running model inference...")
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits
            
            # Resize to original image size
            logits = F.interpolate(
                logits,
                size=image_rgb.shape[:2],
                mode="bilinear",
                align_corners=False
            )
            
            preds = torch.argmax(logits, dim=1).cpu().numpy()[0]
        
        print("✅ Inference completed")
        
        # Get disease mask (class 1)
        disease_mask = (preds == 1).astype(np.uint8)
        
        # Get leaf mask for severity calculation
        print("📏 Calculating severity...")
        leaf_mask = get_leaf_mask(image_rgb)
        
        # Calculate severity
        severity = calculate_severity(leaf_mask, disease_mask)
        
        # Separate disease regions
        print("🔍 Separating disease regions...")
        region_data = separate_disease_regions(disease_mask, image_rgb)
        
        # Create visualization
        print("🎨 Creating visualization...")
        visualization = create_visualization(image_rgb, disease_mask, region_data["markers"])
        
        inference_time = (time.time() - start_time) * 1000
        
        print("✅ All processing completed successfully")
        
        return JSONResponse({
            "success": True,
            "filename": file.filename,
            "inference_time_ms": round(inference_time, 2),
            "severity_percentage": round(severity, 2),
            "disease_pixels": int(np.sum(disease_mask)),
            "leaf_pixels": int(np.sum(leaf_mask)),
            "num_disease_regions": region_data["num_regions"],
            "regions": region_data["regions"],
            "visualization_base64": visualization
        })
    
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/severity")
async def get_severity(file: UploadFile = File(...)):
    """
    Quick severity analysis (returns only severity percentage)
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Prepare input
        inputs = processor(images=image_rgb, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(DEVICE)
        
        # Run inference
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits
            logits = F.interpolate(
                logits,
                size=image_rgb.shape[:2],
                mode="bilinear",
                align_corners=False
            )
            preds = torch.argmax(logits, dim=1).cpu().numpy()[0]
        
        disease_mask = (preds == 1).astype(np.uint8)
        leaf_mask = get_leaf_mask(image_rgb)
        severity = calculate_severity(leaf_mask, disease_mask)
        
        return JSONResponse({
            "success": True,
            "filename": file.filename,
            "severity_percentage": round(severity, 2)
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== RUN SERVER ====================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🌿 Starting Plant Disease Segmentation API")
    print("="*60)
    print(f"📊 Model: SegFormer-B1")
    print(f"💻 Device: {DEVICE}")
    print(f"🌐 Server: http://localhost:8001")
    print(f"📚 Docs: http://localhost:8001/docs")
    print("="*60 + "\n")
    
    uvicorn.run(
        "segment_api:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )