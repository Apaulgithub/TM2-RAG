import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.vision.image_classifier import (
    load_model,
    preprocess_image,
)

from src.config import (
    PATHOLOGIES,
    DATA_DIR,
)

# ============================================================
# RESULT OBJECT
# ============================================================

@dataclass
class GradCAMResult:
    image_path: str
    heatmap_path: str
    overlay_path: str
    target_label: str
    confidence: float


# ============================================================
# GRADCAM
# ============================================================

class GradCAM:
    def __init__(self):
        self.model, self.transforms, self.device = load_model()
        self.model.eval()
        self._disable_inplace_ops()

        self.activations = None
        self.gradients = None
        self._hook_handles = []

        # Target the final convolutional layer for true semantic class activation
        self.target_layer = self._select_target_layer()
        self._register_hooks()

    def _disable_inplace_ops(self):
        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False

    def _select_target_layer(self):
        """
        Selects the final convolutional layer of the DenseNet architecture.
        Grad-CAM works best at the deepest convolutional layer where features 
        are highly class-specific.
        """
        features = self.model.features
        # 'norm5' is the final layer of the feature extractor in DenseNet before pooling
        if hasattr(features, "norm5"):
            print("[GradCAM] Hooking final DenseNet layer: features.norm5")
            return features.norm5
        elif hasattr(features, "denseblock4"):
            print("[GradCAM] Hooking DenseNet layer: features.denseblock4")
            return features.denseblock4
        
        print("[GradCAM] Hooking fallback layer: features")
        return features

    # ========================================================
    # HOOKS
    # ========================================================

    def _register_hooks(self):
            """
            Registers a forward hook that captures activations and hitches a 
            tensor-level backward hook to cleanly extract gradients without 
            triggering PyTorch's in-place view restrictions.
            """
            def forward_hook(module, input, output):
                # 1. Capture activations
                self.activations = output.detach()
                
                # 2. Attach a hook directly to the output tensor to catch its gradient
                def tensor_backward_hook(grad):
                    self.gradients = grad.detach()
                    return grad

                # Using tensor.register_hook bypasses the module-level full_backward_hook bugs
                output.register_hook(tensor_backward_hook)

            self._hook_handles.append(
                self.target_layer.register_forward_hook(forward_hook)
            )

    def _select_target(self, probs, target_label):
        model_labels = list(getattr(self.model, "pathologies", []))

        if target_label:
            candidates = [
                target_label,
                target_label.replace(" ", "_"),
                "Effusion" if target_label == "Pleural Effusion" else None,
                "Pleural_Thickening" if target_label == "Pleural Other" else None,
            ]
            for candidate in candidates:
                if candidate and candidate in model_labels:
                    return model_labels.index(candidate), target_label

        target_idx = int(probs[0].argmax().item())
        if model_labels and target_idx < len(model_labels):
            target_label = model_labels[target_idx]
        elif target_idx < len(PATHOLOGIES):
            target_label = PATHOLOGIES[target_idx]
        else:
            target_label = f"class_{target_idx}"

        return target_idx, target_label

    # ========================================================
    # SINGLE IMAGE
    # ========================================================

    def generate(self, image_path, target_label=None):
            # 1. Read original image first to know its true dimensions
            original = cv2.imread(str(image_path))
            original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            orig_h, orig_w = original.shape[:2]

            # 2. Preprocess to model tensor (typically 224x224)
            img_tensor = preprocess_image(image_path, self.transforms).to(self.device)
            
            # Track what size the tensor actually is inside the model pipeline
            tensor_h, tensor_w = img_tensor.shape[2], img_tensor.shape[3]

            self.gradients = None
            self.activations = None
            self.model.zero_grad(set_to_none=True)

            logits = self.model(img_tensor)
            probs = torch.sigmoid(logits)

            # Target selection
            target_idx, target_label = self._select_target(probs, target_label)
            print(f"[GradCAM] Target class: {target_label} (idx={target_idx})")
            confidence = float(probs[0, target_idx].item())

            # Backward pass
            score = logits[0, target_idx]
            score.backward()

            if self.gradients is None or self.activations is None:
                raise RuntimeError("Grad-CAM hooks did not capture activations/gradients")

            grads = self.gradients[0].cpu().numpy()
            acts = self.activations[0].cpu().numpy()

            # Global average pooling weights
            weights = np.mean(grads, axis=(1, 2))

            # Calculate weighted combination
            cam = np.zeros(acts.shape[1:], dtype=np.float32)
            for i, w in enumerate(weights):
                cam += w * acts[i]

            # ReLU representation
            cam = np.maximum(cam, 0)

            # Safe normalization of raw features
            cam_max = cam.max()
            if cam_max > 0:
                cam = cam / cam_max

            # 3. Match the Model's Preprocessed Canvas Space First
            # Bi-linear interpolation prevents the sharp ringing artifacts (red tracks)
            cam = cv2.resize(cam, (tensor_w, tensor_h), interpolation=cv2.INTER_LINEAR)

            # 4. Map the canvas back to the original image coordinates
            cam = cv2.resize(cam, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            
            # Ensure strict pixel saturation bounds [0, 255]
            heatmap = np.uint8(255 * np.clip(cam, 0, 1))
            overlay = self._overlay(original, heatmap)

            # Zero out heatmap activations that land on pure background padding pixels
            heatmap[cv2.cvtColor(original, cv2.COLOR_RGB2GRAY) < 15] = 0

            return heatmap, overlay, target_label, confidence

    def _overlay(self, image, heatmap):
        # Generate the standard Jet colormap representation
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        # Smooth out alpha blending weights cleanly
        normalized_heat = heatmap.astype(np.float32) / 255.0
        
        # Linear dynamic alpha mapping: higher values look brighter, background remains clear
        alpha = 0.45 * normalized_heat[..., None]

        overlay = (image.astype(np.float32) * (1.0 - alpha) + 
                   heatmap_color.astype(np.float32) * alpha)

        return np.clip(overlay, 0, 255).astype(np.uint8)

    def __del__(self):
        # Clean up hooks upon deletion
        for handle in self._hook_handles:
            handle.remove()


# ============================================================
# MAIN API
# ============================================================

def generate_gradcam(
    image_paths: List[str],
    pathology_labels: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
):
    output_dir = output_dir or DATA_DIR / "gradcam_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    cam = GradCAM()
    results = {}

    for i, image_path in enumerate(image_paths):
        try:
            label = None
            if pathology_labels and i < len(pathology_labels):
                label = pathology_labels[i]

            heatmap, overlay, label, confidence = cam.generate(image_path, label)
            name = Path(image_path).stem

            heatmap_path = output_dir / f"{name}_heatmap.png"
            overlay_path = output_dir / f"{name}_overlay.png"

            cv2.imwrite(str(heatmap_path), heatmap)
            cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

            results[str(image_path)] = GradCAMResult(
                image_path=str(image_path),
                heatmap_path=str(heatmap_path),
                overlay_path=str(overlay_path),
                target_label=label,
                confidence=confidence,
            )
        except Exception as e:
            print(f"[GradCAM] Failed: {image_path}")
            print(e)

    return results