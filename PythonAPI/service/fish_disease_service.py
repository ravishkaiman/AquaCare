import torch
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn
from torchvision.models import ResNet50_Weights

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the validation transform (same as used during training/validation)
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# List of class names in the same order as used during training
CLASS_NAMES = [
    'Bacterial Red disease',
    'Bacterial diseases - Aeromoniasis',
    'Bacterial gill disease',
    'Fungal diseases Saprolegniasis',
    'Healthy Fish',
    'Parasitic diseases',
    'Viral diseases White tail disease'
]

# Global model variable for reuse
model = None


def load_model(model_path: str, num_classes: int = len(CLASS_NAMES)):
    """
    Initialize the model structure, load the saved weights, and set the model to evaluation mode.
    """
    global model
    # Use the recommended weights parameter instead of pretrained=True
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    # Replace the final layer to match our number of classes
    model.fc = nn.Linear(in_features, num_classes)

    # Load the state dictionary
    # Using weights_only=True helps avoid future pickle-related warnings.
    state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()
    return model


def predict_image_from_file(image_file) -> str:
    """
    Given an image file (from a Flask request), preprocess it and return the predicted class name.
    """
    # Open the image and convert to RGB
    image = Image.open(image_file).convert("RGB")
    image_tensor = val_transforms(image)
    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)  # Add batch dimension

    global model
    if model is None:
        raise RuntimeError("Model not loaded. Please call load_model() first.")

    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted_idx = torch.max(outputs, 1)

    return CLASS_NAMES[predicted_idx.item()]
