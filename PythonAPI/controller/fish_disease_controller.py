from flask import Blueprint, request, jsonify

from service.fish_disease_service import load_model, predict_image_from_file

prediction_bp = Blueprint("prediction", __name__)

# Set the path to your saved model weights
MODEL_PATH = "fish_disease_model.pth"

# Load the model once (during blueprint registration)
# You might consider adding error handling in production
load_model(MODEL_PATH)


@prediction_bp.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        prediction = predict_image_from_file(file)
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
