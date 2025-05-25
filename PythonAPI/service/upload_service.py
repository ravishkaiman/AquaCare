import os
import uuid
from werkzeug.utils import secure_filename

class UploadService:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, '..', 'uploads')  # UPLOAD inside project

    @staticmethod
    def save_image(file):
        if not os.path.exists(UploadService.UPLOAD_FOLDER):
            os.makedirs(UploadService.UPLOAD_FOLDER)

        file_ext = os.path.splitext(file.filename)[1]
        unique_name = f"{uuid.uuid4()}{file_ext}"
        filename = secure_filename(unique_name)
        save_path = os.path.join(UploadService.UPLOAD_FOLDER, filename)
        file.save(save_path)
        return filename
