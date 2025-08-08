import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))

from flask import Flask, render_template, request, redirect, url_for
import uuid
from scripts.detect_video import predict_video  # Now it will work!

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'webapp', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return redirect(url_for('index'))

    file = request.files['video']
    if file.filename == '':
        return redirect(url_for('index'))

    filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Run prediction
    label, confidence = predict_video(filepath)
    return render_template('result.html', label=label, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)
