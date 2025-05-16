from flask import Flask, render_template, request, jsonify, send_from_directory, url_for, redirect
import os
import time
import threading
import json
import subprocess
import uuid
import numpy as np
import cv2
import torch
from werkzeug.utils import secure_filename

# Import your PyTorch detection module
from torch_detection import load_violence_detection_model, extract_frames, preprocess_frames, predict_violence

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv'}
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload size
app.config['MODEL_PATH'] = 'model_final.pth'  # Update with your model path
app.config['DETECTION_THRESHOLD'] = 0.6  # Higher threshold to reduce false positives

# Create folders if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Store active jobs and results
active_jobs = {}
results_history = {}

# Load the model at startup
try:
    model, _ = load_violence_detection_model(app.config['MODEL_PATH'])
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_video_metadata(video_path):
    """Get basic metadata from video file"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        # Extract metadata
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        return {
            'width': width,
            'height': height,
            'fps': fps,
            'frame_count': frame_count,
            'duration': duration,
            'duration_formatted': f"{int(duration//60)}:{int(duration%60):02d}"
        }
    except Exception as e:
        print(f"Error getting video metadata: {e}")
        return None

def generate_thumbnail(video_path, output_path, frame_number=0):
    """Generate a thumbnail from the video"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False
        
        # Try to get a frame from the middle of the video
        if frame_number == 0:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count > 0:
                frame_number = frame_count // 4  # Get frame from 25% in
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return False
        
        # Resize for thumbnail
        height, width = frame.shape[:2]
        max_dim = 400
        if height > width:
            new_height = max_dim
            new_width = int(width * (max_dim / height))
        else:
            new_width = max_dim
            new_height = int(height * (max_dim / width))
        
        frame = cv2.resize(frame, (new_width, new_height))
        
        # Save thumbnail
        cv2.imwrite(output_path, frame)
        cap.release()
        return True
    except Exception as e:
        print(f"Error generating thumbnail: {e}")
        return False

def process_video(job_id, video_path, threshold=None):
    """Process a video for violence detection"""
    if model is None:
        active_jobs[job_id]['status'] = 'error'
        active_jobs[job_id]['message'] = 'Model not loaded'
        return
    
    # Use app config threshold if none specified
    if threshold is None:
        threshold = app.config['DETECTION_THRESHOLD']
    
    try:
        # Extract metadata
        metadata = get_video_metadata(video_path)
        if metadata is None:
            active_jobs[job_id]['status'] = 'error'
            active_jobs[job_id]['message'] = 'Could not read video file'
            return
        
        active_jobs[job_id]['metadata'] = metadata
        active_jobs[job_id]['status'] = 'processing'
        active_jobs[job_id]['progress'] = 5
        active_jobs[job_id]['message'] = 'Extracting frames'
        
        # Generate thumbnail
        thumbnail_path = os.path.join(app.config['RESULTS_FOLDER'], f"{job_id}_thumbnail.jpg")
        generate_thumbnail(video_path, thumbnail_path)
        active_jobs[job_id]['thumbnail'] = f"/results/{job_id}_thumbnail.jpg"
        
        # Extract frames
        frames = extract_frames(video_path)
        if frames is None or len(frames) == 0:
            active_jobs[job_id]['status'] = 'error'
            active_jobs[job_id]['message'] = 'Failed to extract frames'
            return
        
        active_jobs[job_id]['progress'] = 30
        active_jobs[job_id]['message'] = 'Preprocessing frames'
        
        # Preprocess frames
        processed_frames = preprocess_frames(frames)
        
        active_jobs[job_id]['progress'] = 50
        active_jobs[job_id]['message'] = 'Running violence detection'
        
        # Make prediction on the full video - enable debug mode
        is_fight, confidence, inference_time = predict_violence(model, processed_frames, threshold, debug=True)
        
        # Process video with Sliding Window approach for a detailed timeline
        segments = []
        window_size = 64  # Number of frames in sliding window
        stride = 16  # Stride between windows
        
        total_frames = len(frames)
        duration = metadata['duration']
        
        # For a proper timeline, analyze multiple windows if the video is long enough
        if total_frames > window_size:
            window_count = (total_frames - window_size) // stride + 1
            
            for i in range(window_count):
                active_jobs[job_id]['progress'] = 50 + int(40 * i / window_count)
                
                start_idx = i * stride
                end_idx = start_idx + window_size
                
                # Extract window frames and preprocess
                window_frames = frames[start_idx:end_idx]
                if len(window_frames) == window_size:
                    # Preprocess this window
                    window_tensor = preprocess_frames(window_frames)
                    
                    # Higher confidence threshold for segments to reduce false positives
                    segment_threshold = threshold + 0.1  # More strict for segments
                    is_violent, prob, _ = predict_violence(model, window_tensor, segment_threshold, debug=False)
                    
                    if is_violent and prob > segment_threshold:
                        # Convert frame indices to timestamps
                        start_time = (start_idx / total_frames) * duration
                        end_time = (end_idx / total_frames) * duration
                        
                        segments.append({
                            'start': start_time,
                            'end': end_time,
                            'confidence': float(prob),
                            'start_formatted': f"{int(start_time//60)}:{int(start_time%60):02d}",
                            'end_formatted': f"{int(end_time//60)}:{int(end_time%60):02d}"
                        })
        else:
            # For very short videos, just use the overall result
            if is_fight and confidence > threshold:
                segments.append({
                    'start': 0,
                    'end': duration,
                    'confidence': float(confidence),
                    'start_formatted': "0:00",
                    'end_formatted': metadata['duration_formatted']
                })
        
        # Merge overlapping segments
        if segments:
            merged_segments = [segments[0]]
            for segment in segments[1:]:
                prev = merged_segments[-1]
                if segment['start'] <= prev['end']:
                    # Merge overlapping segments
                    prev['end'] = max(prev['end'], segment['end'])
                    prev['confidence'] = max(prev['confidence'], segment['confidence'])
                    prev['end_formatted'] = f"{int(prev['end']//60)}:{int(prev['end']%60):02d}"
                else:
                    merged_segments.append(segment)
            segments = merged_segments
        
        # Save final results
        result = {
            'job_id': job_id,
            'video_path': video_path,
            'filename': os.path.basename(video_path),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'metadata': metadata,
            'thumbnail': active_jobs[job_id]['thumbnail'],
            'overall_result': {
                'is_fight': is_fight,
                'confidence': float(confidence),
                'inference_time': inference_time
            },
            'segments': segments,
            'has_violence': len(segments) > 0
        }
        
        # Save result to JSON file
        result_path = os.path.join(app.config['RESULTS_FOLDER'], f"{job_id}_result.json")
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Update active job and add to history
        active_jobs[job_id]['status'] = 'completed'
        active_jobs[job_id]['progress'] = 100
        active_jobs[job_id]['message'] = 'Processing complete'
        active_jobs[job_id]['result'] = result
        
        # Add to history
        results_history[job_id] = {
            'job_id': job_id,
            'filename': os.path.basename(video_path),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'has_violence': len(segments) > 0,
            'thumbnail': active_jobs[job_id]['thumbnail']
        }
        
        # Save history to file
        with open(os.path.join(app.config['RESULTS_FOLDER'], 'history.json'), 'w') as f:
            json.dump(list(results_history.values()), f, indent=2)
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        active_jobs[job_id]['status'] = 'error'
        active_jobs[job_id]['message'] = f'Error: {str(e)}'

@app.route('/')
def index():
    """Render the dashboard"""
    # Load history from file if it exists
    history_path = os.path.join(app.config['RESULTS_FOLDER'], 'history.json')
    if os.path.exists(history_path):
        try:
            with open(history_path, 'r') as f:
                history = json.load(f)
                for item in history:
                    results_history[item['job_id']] = item
        except Exception as e:
            print(f"Error loading history: {e}")
    
    return render_template('index.html', 
                           active_jobs=active_jobs,
                           history=list(results_history.values()))

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    if 'file' not in request.files and 'videoPath' not in request.form:
        return jsonify({
            'success': False,
            'message': 'No file or path provided'
        }), 400
    
    # Generate a job ID
    job_id = str(uuid.uuid4())
    
    if 'file' in request.files and request.files['file'].filename != '':
        # Handle file upload
        file = request.files['file']
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'message': 'Invalid file type'
            }), 400
        
        # Save the file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}_{filename}")
        file.save(file_path)
        
        # Create job
        active_jobs[job_id] = {
            'id': job_id,
            'file_path': file_path,
            'filename': filename,
            'status': 'queued',
            'progress': 0,
            'message': 'Queued for processing',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Start processing in a thread
        threading.Thread(target=process_video, args=(job_id, file_path)).start()
        
        return jsonify({
            'success': True,
            'message': 'File uploaded successfully',
            'job_id': job_id
        })
    
    elif 'videoPath' in request.form and request.form['videoPath'].strip():
        # Handle local path
        video_path = request.form['videoPath'].strip()
        if not os.path.exists(video_path):
            return jsonify({
                'success': False,
                'message': 'File not found at specified path'
            }), 400
        
        # Check if it's a valid video file
        _, ext = os.path.splitext(video_path)
        if ext.lower()[1:] not in app.config['ALLOWED_EXTENSIONS']:
            return jsonify({
                'success': False,
                'message': 'Invalid file type'
            }), 400
        
        # Create job
        active_jobs[job_id] = {
            'id': job_id,
            'file_path': video_path,
            'filename': os.path.basename(video_path),
            'status': 'queued',
            'progress': 0,
            'message': 'Queued for processing',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Start processing in a thread
        threading.Thread(target=process_video, args=(job_id, video_path)).start()
        
        return jsonify({
            'success': True,
            'message': 'Video path submitted successfully',
            'job_id': job_id
        })
    
    return jsonify({
        'success': False,
        'message': 'No file or valid path provided'
    }), 400

@app.route('/job/<job_id>')
def get_job(job_id):
    """Get job status"""
    if job_id not in active_jobs:
        return jsonify({
            'success': False,
            'message': 'Job not found'
        }), 404
    
    return jsonify({
        'success': True,
        'job': active_jobs[job_id]
    })

@app.route('/result/<job_id>')
def get_result(job_id):
    """Get detailed result page for a job"""
    # Check if result exists
    result_path = os.path.join(app.config['RESULTS_FOLDER'], f"{job_id}_result.json")
    if not os.path.exists(result_path):
        return render_template('error.html', message='Result not found')
    
    # Load result
    with open(result_path, 'r') as f:
        result = json.load(f)
    
    return render_template('result.html', result=result)

@app.route('/results')
def get_results():
    """Get all results for the history page"""
    return render_template('history.html', history=list(results_history.values()))

@app.route('/results/<path:filename>')
def results_files(filename):
    """Serve files from the results folder"""
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)