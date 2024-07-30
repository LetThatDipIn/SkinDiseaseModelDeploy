from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
from tensorflow.keras.applications.resnet50 import preprocess_input

app = FastAPI()

# Load your model
model = load_model("ResNet50_model.h5")

# Define your class labels
class_labels = ['Acne and Rosacea Photos', 'Atopic Dermatitis Photos', 'Basal Cell Carcinoma (BCC)', 'Benign Keratosis-like Lesions (BKL)', 'Melanocytic Nevi (NV)', 'Nail Fungus and other Nail Disease']

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')  # Ensure image is in RGB mode
    
    # Preprocess the image
    image = image.resize((224, 224))  # Example size
    image_array = np.array(image)  # Convert image to array
    image_array = np.expand_dims(image_array, axis=0)  # Expand dimensions
    image_array = preprocess_input(image_array)  # Apply preprocessing

    # Make prediction
    prediction = model.predict(image_array)

    # Find the index of the highest probability
    predicted_class_index = np.argmax(prediction[0])
    predicted_class_label = class_labels[predicted_class_index]
    
    result = {
        "prediction": prediction.tolist(),
        "predicted_class": predicted_class_label
    }
    return result

@app.get("/")
async def root():
    return {"message": "Skin Disease Classification API"}