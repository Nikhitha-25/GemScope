from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import sqlite3
from predict import predict_gemstone_name  # This should be your image prediction function

app = Flask(__name__)

# Upload folder for storing uploaded images
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def get_gem_details(gem_name):
    db_path = os.path.join(os.getcwd(), 'final_gemstones_db.db')  # Correct DB file
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM final_gemstone_data WHERE name = ? COLLATE NOCASE", (gem_name,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No image uploaded"

    file = request.files['image']
    if file.filename == '':
        return "No selected file"

    # Save the uploaded image
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Predict gemstone name from image path
    gem_name = predict_gemstone_name(filepath)
    print("Predicted Gemstone:", gem_name)

    # Get gem details from database
    gem_details = get_gem_details(gem_name)
    if not gem_details:
        return render_template('result.html', prediction=gem_name,
                               image_path=os.path.join('static', 'uploads', filename).replace("\\", "/"),
                               error="Gemstone details not found in the database.")

    # Construct relative path for displaying image in frontend
    relative_path = os.path.join('static', 'uploads', filename).replace("\\", "/")

    return render_template('result.html', prediction=gem_name,
                           image_path=relative_path,
                           gem=gem_details)


if __name__ == '__main__':
    app.run(debug=True)
