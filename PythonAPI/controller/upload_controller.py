import os
from flask import Blueprint, request, jsonify, send_from_directory, abort

from service.upload_service import UploadService

upload_bp = Blueprint('upload', __name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, '..', 'uploads')

@upload_bp.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = UploadService.save_image(file)
    return jsonify({'filename': filename, 'download_url': f'/api/uploads/{filename}'}), 200

@upload_bp.route('/uploads/<filename>', methods=['GET'])
def download_image(filename):
    file_path = os.path.join(UPLOAD_FOLDER, filename)

    if not os.path.exists(file_path):
        print(f"[ERROR] File not found for download: {file_path}")  # 👈 print to console
        abort(404, description="File not found.")

    print(f"[INFO] File found, sending: {file_path}")  # 👈 print successful download path
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)
