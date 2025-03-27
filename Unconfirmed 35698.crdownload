from flask import Flask, request, jsonify, render_template
import whisper
import pickle
import os
from flask_cors import CORS
from pyngrok import ngrok

# Load Whisper Model
whisper_model = whisper.load_model("tiny")  # Use "tiny" for faster response

# Load Scam Detection Model & Vectorizer
with open("scam_detection_model.pkl", "rb") as model_file:
    scam_model = pickle.load(model_file)

with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Initialize Flask App
app = Flask(__name__, template_folder="templates")
CORS(app)

@app.route("/")
def home():
    return render_template("index.html")  # Serve HTML file

@app.route("/predict", methods=["POST"])
def predict_scam():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Ensure the form-data key is 'file'."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty file name, please upload a valid file."}), 400

    filepath = "uploaded_audio.wav"
    file.save(filepath)

    try:
        # Transcribe audio using Whisper
        result = whisper_model.transcribe(filepath)
        transcribed_text = result["text"]

        # Predict Scam Type
        text_features = vectorizer.transform([transcribed_text])
        predicted_scam = scam_model.predict(text_features)

        return jsonify({"transcription": transcribed_text, "scam_type": int(predicted_scam[0])})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

if __name__ == "__main__":
    public_url = ngrok.connect(5000).public_url
    print("Ngrok tunnel URL:", public_url)
    app.run(host="0.0.0.0", port=5000)
