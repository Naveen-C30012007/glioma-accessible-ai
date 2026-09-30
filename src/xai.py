import os
import torch
import torch.nn as nn
from torchvision import models
import numpy as np

# Path resolution
PRIMARY_MODEL_PATH = r"C:\Users\Naveen\glioma-accessible-ai\models\glioma_model.pth"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SECONDARY_MODEL_PATH = os.path.join(BASE_DIR, "models", "glioma_model.pth")
MODEL_PATH = PRIMARY_MODEL_PATH if os.path.exists(PRIMARY_MODEL_PATH) else SECONDARY_MODEL_PATH

def load_trained_model(model_path=MODEL_PATH):
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor):
        output = self.model(input_tensor)
        class_idx = torch.argmax(output, dim=1).item()
        
        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        gradients = self.gradients.data.numpy()[0]
        activations = self.activations.data.numpy()[0]

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        cam = np.maximum(cam, 0)
        if np.max(cam) > 0:
            cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
        
        probabilities = torch.softmax(output, dim=1).detach().numpy()[0]
        return cam, class_idx, probabilities

# --- Plain-Language Explanation Generator ---
def generate_plain_language_report(class_idx, confidence, heatmap):
    # PyTorch ImageFolder alphabetical index mapping:
    # Class 0: 'glioma'
    # Class 1: 'healthy' / 'notumor'
    labels = ["Glioma Detected", "Healthy Control (No Tumour Identified)"]
    predicted_label = labels[class_idx]
    
    high_activation_ratio = np.sum(heatmap > 0.6) / heatmap.size
    
    report = []
    report.append("=== ACCESSIBLE MRI ANALYSIS REPORT ===")
    report.append(f"Classification : {predicted_label}")
    report.append(f"AI Confidence  : {confidence * 100:.1f}%\n")
    report.append("Plain-Language Observations:")
    
    if class_idx == 0:  # Glioma Case
        report.append("• Structural Anomaly: The AI highlighted tissue texture variations characteristic of abnormal growth.")
        
        if high_activation_ratio > 0.15:
            report.append("• Suspicious Area: The flagged region covers a broader localized segment of the scan.")
        else:
            report.append("• Suspicious Area: The flagged region is confined to a small focal zone.")
            
        if confidence >= 0.85:
            report.append("• Pattern Match: Strong alignment with documented glioma MRI features.")
        else:
            report.append("• Pattern Match: Moderate alignment; clinical evaluation is strongly recommended.")
            
        report.append("\nNext Steps:")
        report.append("This AI tool provides exploratory screening. Please share this report with a medical professional for confirmation.")
    else:  # Healthy Case
        report.append("• Structural Assessment: Tissue patterns appear uniform without localized cell growth anomalies.")
        report.append("• Pattern Match: Scan features closely align with healthy control references.")
        report.append("\nNext Steps:")
        report.append("No immediate tumor-related indicators detected.")

    return "\n".join(report)