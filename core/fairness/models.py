# moq-nas/core/fairness/models.py

import torch.nn as nn
from torchvision import models
from typing import Callable, Dict, Tuple, Optional

# --- Helper functions to modify model heads ---

def _set_fc(m, nc):
    """Replaces the 'fc' layer of a model like ResNet."""
    m.fc = nn.Linear(m.fc.in_features, nc)

def _replace_last_linear_in_sequential(seq: nn.Sequential, nc: int):
    """Finds and replaces the last linear layer in a sequential classifier."""
    idx = None
    for i in reversed(range(len(seq))):
        if isinstance(seq[i], nn.Linear):
            idx = i
            break
    if idx is None:
        raise RuntimeError("Classifier tail has no Linear layer; could not replace.")
    
    in_features = seq[idx].in_features
    seq[idx] = nn.Linear(in_features, nc)

def _set_cls(model, nc):
    """Replaces the head of models with a '.classifier' sequential attribute."""
    _replace_last_linear_in_sequential(model.classifier, nc)


def _set_convnext_head(m, nc):
    """Replaces the head of a ConvNeXt model."""
    lin = m.classifier[-1]
    m.classifier[-1] = nn.Linear(lin.in_features, nc)


# --- Model Registry ---
# Maps an architecture name to its torchvision constructor, pre-trained weights, 
# and the correct head-replacement function defined above.

REGISTRY: Dict[str, Tuple[Callable, object, Callable]] = {}
try: REGISTRY["resnet18"] = (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1, _set_fc)
except Exception: pass
try: REGISTRY["resnet50"] = (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2, _set_fc)
except Exception: pass
try: REGISTRY["mobilenet_v3_large"] = (models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights.IMAGENET1K_V1, _set_cls)
except Exception: pass
try: REGISTRY["efficientnet_v2_s"] = (models.efficientnet_v2_s, models.EfficientNet_V2_S_Weights.IMAGENET1K_V1, _set_cls)
except Exception: pass
try: REGISTRY["convnext_tiny"] = (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1, _set_convnext_head)
except Exception: pass


# --- Main Factory Function ---

def make_baseline_model(arch: str, num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """
    Builds a torchvision model. Can optionally load ImageNet weights or initialize 
    from scratch, then replaces the final classification layer.

    Args:
        arch (str): The name of the architecture (e.g., 'resnet18').
        num_classes (int): The number of output classes.
        pretrained (bool): If True, load ImageNet weights. If False, train from scratch.

    Returns:
        torch.nn.Module: The ready-to-train baseline model.
    """
    if arch not in REGISTRY:
        raise ValueError(f"Architecture '{arch}' is not in the registry. "
                        f"Available models: {list(REGISTRY.keys())}")

    ctor, weights_enum, head_setter = REGISTRY[arch]

    # Instantiate the model
    if pretrained:
        # Load Pre-trained ImageNet Weights
        try:
            weights = getattr(weights_enum, "DEFAULT", weights_enum)
            model = ctor(weights=weights)
        except Exception:
            # Fallback for older torchvision versions
            model = ctor(pretrained=True)
    else:
        # Initialize Randomly (From Scratch)
        try:
            model = ctor(weights=None)
        except Exception:
            # Fallback for older torchvision versions
            model = ctor(pretrained=False)
    
    # Replace the final layer for your binary classification task
    # This must happen regardless of initialization to ensure output dims are correct
    head_setter(model, num_classes)
    
    return model