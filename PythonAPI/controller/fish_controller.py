# fish_controller.py

from flask import Blueprint, request, jsonify

from service.fish_service import predict_image

fish_bp = Blueprint("fish", __name__)


@fish_bp.route("/predict", methods=["POST"])
def predict():
    # 1) check file
    if "image" not in request.files:
        return jsonify({"error": "No image file found in request"}), 400

    img = request.files["image"]
    if img.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # 2) delegate prediction
    try:
        label = predict_image(img)
        return jsonify({"predicted_class": label}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
