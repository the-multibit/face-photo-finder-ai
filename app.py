import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import face_recognition
import numpy as np
from werkzeug.utils import secure_filename # For safe file names

app = Flask(__name__)

# --- Configuration ---
# Set the path to the folder containing your family photos.
# IMPORTANT: Replace 'path/to/your/family_photos' with the actual path on your computer.
# This folder will be 'exposed' as a static directory for the web app.
FAMILY_PHOTOS_FOLDER = 'YOUR_ALBUM_PATH_HERE' # Update this!

# Folder to temporarily store uploaded face images
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Configure Flask to serve files from your family photos folder
# This makes images in FAMILY_PHOTOS_FOLDER accessible via /family_photos_static/filename.jpg
app.add_url_rule(
    '/family_photos_static/<path:filename>',
    endpoint='family_photos_static',
    view_func=lambda filename: send_from_directory(FAMILY_PHOTOS_FOLDER, filename)
)


# --- Helper Functions ---
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_and_encode_face(image_path):
    """
    Loads an image and encodes the first face found in it.
    Returns the face encoding if successful, None otherwise.
    """
    try:
        image = face_recognition.load_image_file(image_path)
        face_encodings = face_recognition.face_encodings(image)
        if face_encodings:
            return face_encodings[0]
        else:
            return None
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def find_my_photos(known_face_encoding, family_photos_folder):
    """
    Scans a folder of photos and identifies which ones contain the known face.
    Returns a list of filenames where a match is found.
    """
    if not os.path.isdir(family_photos_folder):
        print(f"Error: Family photos folder not found at {family_photos_folder}")
        return []

    found_in_photos = []
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')

    for filename in os.listdir(family_photos_folder):
        if filename.lower().endswith(image_extensions):
            image_path = os.path.join(family_photos_folder, filename)
            try:
                unknown_image = face_recognition.load_image_file(image_path)
                face_locations = face_recognition.face_locations(unknown_image)
                unknown_face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

                if not unknown_face_encodings:
                    continue

                for face_encoding in unknown_face_encodings:
                    matches = face_recognition.compare_faces([known_face_encoding], face_encoding, tolerance=0.5)
                    if True in matches:
                        found_in_photos.append(filename)
                        break
            except Exception as e:
                print(f"Could not process {filename}: {e}")
                continue
    return found_in_photos

# --- Flask Routes ---

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    found_photos = []
    message = None
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            message = 'No file part'
            return render_template('index.html', found_photos=found_photos, message=message)
        
        file = request.files['file']
        
        # If the user does not select a file, the browser submits an empty file without a filename.
        if file.filename == '':
            message = 'No selected file'
            return render_template('index.html', found_photos=found_photos, message=message)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path)
            
            # Process the uploaded face
            my_face_encoding = load_and_encode_face(upload_path)
            
            if my_face_encoding is not None:
                found_photos = find_my_photos(my_face_encoding, FAMILY_PHOTOS_FOLDER)
                if not found_photos:
                    message = "No photos found containing your face."
            else:
                message = "No face detected in the uploaded image. Please try again with a clear photo."
            
            # Clean up the uploaded file
            os.remove(upload_path)

    return render_template('index.html', found_photos=found_photos, message=message)

# --- Run the App ---
if __name__ == '__main__':
    # Create the uploads folder if it doesn't exist
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True) # debug=True allows for automatic reloading on code changes
