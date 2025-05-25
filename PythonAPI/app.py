from flask import Flask, Blueprint

from controller.fish_controller import fish_bp
from controller.fish_disease_controller import prediction_bp
from controller.upload_controller import upload_bp

app = Flask(__name__)

# Register the prediction blueprint on the /api route prefix
app.register_blueprint(prediction_bp, url_prefix="/api")
app.register_blueprint(upload_bp, url_prefix="/api")
app.register_blueprint(fish_bp, url_prefix="")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
