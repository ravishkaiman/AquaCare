# fish_service.py

import os
import tempfile
import torch
import torch.nn as nn
from torchvision import transforms, models, datasets
from PIL import Image

class FishClassifier(nn.Module):
    def __init__(self, num_classes: int = 31):
        super().__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

# --- Device & model setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# instantiate and load
_model = FishClassifier().to(device)
_model.load_state_dict(torch.load("fish_classifier_model.pth", map_location=device))
_model.eval()

# load class names from train folder
_train_ds = datasets.ImageFolder("train", transform=transforms.ToTensor())
class_names = _train_ds.classes

# image preprocessing pipeline
_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225],
    ),
])

def predict_image(file_storage) -> str:
    """
    Accepts a Werkzeug FileStorage object (from request.files['image']),
    saves it temporarily, runs through the model, and returns the label.
    """
    # write to temp file
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, file_storage.filename)
    file_storage.save(temp_path)

    try:
        img = Image.open(temp_path).convert("RGB")
        tensor = _transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = _model(tensor)
            _, idx = torch.max(outputs, 1)
            return class_names[idx.item()]
    finally:
        # cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.isdir(temp_dir):
            os.rmdir(temp_dir)
