from flask import Flask, render_template, request, redirect, url_for, jsonify, flash
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import cv2
import os
import numpy as np
import uuid
import time
import logging

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'webp', 'gif'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create folders if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Load the model
try:
    model = YOLO('runs/detect/yolov8_fruit/weights/best.pt')
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def count_fruits(image_path, model, conf=0.25):
    """
    Runs inference on the image, counts fruits per class, and returns both the counts and results.
    """
    try:
        # Run inference
        results = model.predict(image_path, conf=conf)[0]

        # Initialize fruit counter dictionary
        fruit_counts = {}

        # Iterate over each detected box
        for box in results.boxes:
            class_id = int(box.cls)
            class_name = model.names[class_id]
            confidence = float(box.conf)
            
            # Handle different formats of class names
            if '-' in class_name:
                # If class name already includes calories
                class_with_confidence = class_name
            else:
                # Default format if calories not included in name
                class_with_confidence = f"{class_name}"
                
            fruit_counts[class_with_confidence] = fruit_counts.get(class_with_confidence, 0) + 1

        return fruit_counts, results, None
    except Exception as e:
        logger.error(f"Error in fruit detection: {str(e)}")
        return {}, None, str(e)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    try:
        # Generate a unique filename
        unique_filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        
        # Return the filename so the frontend can use it for the next request
        return jsonify({
            'status': 'success',
            'filename': unique_filename,
            'message': 'File uploaded successfully'
        })
        
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/process', methods=['POST'])
def process_image():
    data = request.json
    filename = data.get('filename')
    conf_threshold = float(data.get('confidence', 0.25))
    
    if not filename:
        return jsonify({'error': 'No filename provided'}), 400
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404
    
    try:
        # Process the image
        start_time = time.time()
        counts, results, error = count_fruits(file_path, model, conf=conf_threshold)
        
        if error:
            return jsonify({'error': error}), 500
        
        # Generate and save result image
        results_image = results.plot()  # This returns a numpy array with the detections drawn
        result_filename = f"result_{filename}"
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        
        # Save the result image
        cv2.imwrite(result_path, results_image)
        
        # Get processing stats from results
        processing_time = results.speed
        total_time = time.time() - start_time
        
        return jsonify({
            'status': 'success',
            'counts': counts,
            'original_image': filename,
            'result_image': result_filename,
            'processing_time': {
                'preprocess': processing_time.get('preprocess', 'N/A'),
                'inference': processing_time.get('inference', 'N/A'),
                'postprocess': processing_time.get('postprocess', 'N/A'),
                'total': round(total_time * 1000, 2)
            }
        })
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Health check endpoint
@app.route('/health')
def health_check():
    return jsonify({'status': 'ok', 'model_loaded': model is not None})

if __name__ == '__main__':
    app.run(debug=True)