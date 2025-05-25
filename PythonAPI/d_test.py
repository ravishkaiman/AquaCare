import torch
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the same transform as in the validation phase
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# List the class names in the same order as used during training
class_names = [
    'Bacterial Red disease',
    'Bacterial diseases - Aeromoniasis',
    'Bacterial gill disease',
    'Fungal diseases Saprolegniasis',
    'Healthy Fish',
    'Parasitic diseases',
    'Viral diseases White tail disease'
]


def load_model(model_path, num_classes):
    """
    Initialize the model structure, load the saved weights, and set the model to evaluation mode.
    """
    model = models.resnet50(pretrained=True)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def predict_image(image_path, model):
    """
    Load an image, preprocess it, and run it through the model to obtain a predicted class.
    """
    image = Image.open(image_path).convert('RGB')
    image_tensor = val_transforms(image)
    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)  # Add batch dimension

    with torch.no_grad():
        outputs = model(image_tensor)
        # Get the index of the class with highest probability
        _, predicted_idx = torch.max(outputs, 1)

    return class_names[predicted_idx.item()]


if __name__ == '__main__':
    # Path to the saved model weights
    model_path = "best_fish_disease_model.pth"  # Update if necessary

    # Load the model
    model = load_model(model_path, num_classes=len(class_names))

    # Path to a test image file (update with a valid path)
    test_image_path = "image.jpeg"  # e.g., "data/test/fish_sample.jpg"

    # Predict the class of the test image
    predicted_class = predict_image(test_image_path, model)
    print("Predicted class:", predicted_class)
