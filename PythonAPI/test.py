import os
import tempfile
from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from torchvision import transforms, models, datasets
from PIL import Image

class FishClassifier(nn.Module):
    def __init__(self, num_classes=31):
        super(FishClassifier, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 31
model = FishClassifier(num_classes=num_classes).to(device)

model_path = "fish_classifier_model.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()



train_dataset = datasets.ImageFolder("train", transform=transforms.ToTensor())
class_names = train_dataset.classes


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])


app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return "Welcome to the Fish Classifier API!"

@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({"error": "No image file found in request"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, file.filename)
    file.save(temp_path)

    try:

        image = Image.open(temp_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted_class = torch.max(outputs, 1)
            pred_idx = predicted_class.item()
            pred_label = class_names[pred_idx]


        response = {
            "predicted_class": pred_label
        }
        return jsonify(response), 200

    finally:

        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
