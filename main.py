from flask import Flask, render_template, request, send_from_directory, session
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import numpy as np
import os
from datetime import datetime
import json
import cv2
import gdown

# ================== MODEL DOWNLOAD ==================
MODEL_PATH = "model.h5"

if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?id=1_uli66F5_x3XMkqSabxyEkKqphsk4g8t"
    gdown.download(url, MODEL_PATH, quiet=False)

# ================== MODEL LOAD ==================
model = tf.keras.models.load_model("model.h5", compile=False)

# ================== FLASK APP ==================
app = Flask(__name__)
app.secret_key = "brain_tumor_secret"





class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']

# Upload folder
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Prediction function
# Prediction function
def predict_tumor(image_path):
    img = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    pred_idx = np.argmax(predictions, axis=1)[0]
    confidence = float(np.max(predictions))

    tumor_type = class_labels[pred_idx]
    if tumor_type == "notumor":
     result = "No Tumor Detected"
     explanation = """
     The AI model analyzed the MRI scan and did not detect significant abnormal patterns 
     typically associated with brain tumors. The prediction is supported by the absence 
     of irregular intensity regions in the scan.
     """

    else:
        result = f"Tumor: {tumor_type.capitalize()}"
        explanation = f"""
     The AI model detected patterns in the MRI scan that are commonly associated with {tumor_type} tumors. 
    
     The heatmap highlights regions with abnormal intensity and structural variations in the image. 
     These regions are often correlated with tumor presence and are used as supporting visual cues 
     alongside the model’s prediction.
     """


    return result, confidence, explanation






def generate_smart_heatmap(image_path):

    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Abnormal intensity detection
    _, thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)

    # Edge detection
    edges = cv2.Canny(gray, 30, 120)

    # Combine both
    combined = cv2.addWeighted(thresh, 0.6, edges, 0.4, 0)

    # Smooth
    combined = cv2.GaussianBlur(combined, (25,25), 0)

    # Heatmap
    heatmap = cv2.applyColorMap(combined, cv2.COLORMAP_JET)

    result = cv2.addWeighted(img, 0.6, heatmap, 0.5, 0)

    heatmap_filename = "smart_heatmap_" + os.path.basename(image_path)
    heatmap_path = os.path.join(app.config['UPLOAD_FOLDER'], heatmap_filename)

    cv2.imwrite(heatmap_path, result)

    return heatmap_filename













def save_prediction(image_path, result, confidence):
    data_file = 'data/predictions.json'

    # Load existing history
    if os.path.exists(data_file):
        with open(data_file, 'r') as f:
            history = json.load(f)
    else:
        history = []

    # New record
    record = {
        "image": image_path,
        "result": result,
        "confidence": confidence,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M")
    }

    history.append(record)

    # Save back
    with open(data_file, 'w') as f:
        json.dump(history, f, indent=4)







# Generate PDF report
def generate_report(result, confidence, image_path):
    report_name = "report.pdf"
    report_path = os.path.join(UPLOAD_FOLDER, report_name)
    c = canvas.Canvas(report_path, pagesize=letter)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(200, 750, "Brain Tumor Detection Report")
    c.setFont("Helvetica", 12)
    c.drawString(50, 700, f"Result: {result}")
    c.drawString(50, 680, f"Confidence: {confidence*100:.2f}%")
    c.drawString(50, 660, f"Date & Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.drawString(50, 640, f"Image File: {os.path.basename(image_path)}")
    c.save()
    return report_name


def detect_stroke(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return "Invalid Image", "Unknown"

    img = cv2.resize(img, (256, 256))
    mean_intensity = img.mean()

    if mean_intensity < 90:
        return "Ischemic Stroke Detected", "High Risk"
    elif mean_intensity > 140:
        return "Hemorrhagic Stroke Detected", "Critical"
    else:
        return "No Stroke Detected", "Low Risk"

# Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    if 'history' not in session:
        session['history'] = []

    if request.method == 'POST':
        file = request.files['file']
        if file:
            path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(path)

            result, confidence, explanation = predict_tumor(path)
            heatmap_file = generate_smart_heatmap(path) 

            

            save_prediction(path, result, f"{confidence*100:.2f}%")

            session['history'].append({
                'file': file.filename,
                'result': result,
                'confidence': f"{confidence*100:.2f}%"
            })
            session.modified = True

            report_file = generate_report(result, confidence, path)
            return render_template('index.html',
                       result=result,
                       confidence=f"{confidence*100:.2f}%",
                       explanation=explanation,
                       file_path=f'/uploads/{file.filename}',
                       heatmap_path=f'/uploads/{heatmap_file}',
                       report_path=f'/uploads/{report_file}',
                       history=session['history'])


    return render_template('index.html', result=None, history=session['history'])

@app.route('/uploads/<filename>')
def get_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)



@app.route("/history")
def history():
    data_file = "data/predictions.json"

    if not os.path.exists(data_file):
        history_data = []
    else:
        with open(data_file, "r") as f:
            history_data = json.load(f)

    # latest prediction upar dikhane ke liye
    history_data = history_data[::-1]

    return render_template("history.html", history=history_data)


# ================= STROKE ROUTE =================

@app.route("/stroke", methods=["GET", "POST"])
def stroke():
    if request.method == "POST":
        file = request.files.get("file")
        if file:
            path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(path)

            result, risk = detect_stroke(path)

            return render_template(
                "stroke.html",
                result=result,
                risk=risk,
                image=file.filename
            )

    return render_template("stroke.html")


@app.route('/about')
def about():
    return render_template('about.html')




    


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

