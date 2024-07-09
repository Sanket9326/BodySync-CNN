from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

app = Flask(__name__)

MODEL_PATHS = {
    "model1": "models/brain_tumor_classification_model.h5",
    "model2": "models/breast_cancer_model.h5",
    "model3": "models/kidney_tumor_classification_model.h5",
    "model4": "models/lung_cancer_model.h5",
    "model5": "models/oral_cancer_model.h5",
}

models = {}
for key, path in MODEL_PATHS.items():
    models[key] = load_model(path)

def predict_tumor(model, image):
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=1)
    return predicted_class

def predict_lung_tumor(model, image):
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=1)
    return predicted_class

def preprocess_brain_image(image_file, target_size=(150, 150)):
    img = load_img(image_file, target_size=target_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def preprocess_image(image_path, target_size=(224, 224)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def preprocess_lung_image(image_file, target_size=(150, 150)): 
    img = load_img(image_file, target_size=target_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def preprocess_kidney_image(image_file, target_size=(150, 150)):
    img = load_img(image_file, target_size=target_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route("/analyze", methods=["POST"])
def analyze_tumor():
    selected_category = request.form.get("selected_category")
    image_file = request.files["image"]

    try:
        image_path = "temp_image.jpg"
        image_file.save(image_path)

        if selected_category == "Brain Tumor":
            preprocessed_img = preprocess_brain_image(image_path)
            prediction = predict_tumor(models["model1"], preprocessed_img)
            predicted_class = ["Glioma", "Meningioma", "No Tumor", "Pituitary"][prediction[0]]
        elif selected_category == "Lung Tumor":
            preprocessed_img = preprocess_lung_image(image_path)
            prediction = predict_lung_tumor(models["model4"], preprocessed_img)
            predicted_class = ["Benign", "Malignant", "Normal"][prediction[0]]
        elif selected_category == "Kidney Tumor":
            preprocessed_img = preprocess_kidney_image(image_path)
            prediction = predict_tumor(models["model3"], preprocessed_img)
            predicted_class = ["Benign", "Malignant", "Normal"][prediction[0]]
        elif selected_category in ["Breast Tumor", "Oral Tumor"]:
            preprocessed_img = preprocess_image(image_path)
            prediction = predict_tumor(models["model2"], preprocessed_img)
            predicted_class = "Tumor Present" if prediction[0] == 1 else "No Tumor"
        else:
            return jsonify({"error": "Invalid category selected"})

        return jsonify(
            {
                "selected_category": selected_category,
                "image_filename": image_file.filename,
                "prediction": predicted_class,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)})
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
