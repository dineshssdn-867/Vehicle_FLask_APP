import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from main import Model

UPLOAD_FOLDER = 'static/uploads/'

vehicle_count_app = Flask(__name__)
vehicle_count_app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@vehicle_count_app.route("/")
@vehicle_count_app.route("/home")
def home():
    return render_template("index.html")


@vehicle_count_app.route('/predict_count_vehicles_in_video', methods=['POST'])
def predict_count_vehicles_in_video():
    if request.method == 'POST':
        if 'video' not in request.files:
            return redirect(request.referrer)
        file = request.files['video']
        filename = secure_filename(file.filename)
        file.save(os.path.join(vehicle_count_app.config['UPLOAD_FOLDER'], filename))
        model = Model(os.path.join(vehicle_count_app.config['UPLOAD_FOLDER'], filename))
        csv_name, video_name = model.realTime()
        return render_template('index.html', filename=video_name)

if __name__ == "__main__":
    vehicle_count_app.secret_key = 'super secret key'
    vehicle_count_app.run(debug=True)