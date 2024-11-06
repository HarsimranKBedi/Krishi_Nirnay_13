from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import io
from pydantic import BaseModel
import pickle
import numpy as np
from typing import List, Dict, Optional
from fastapi.middleware.cors import CORSMiddleware
from app.data.disease_data import disease_data  

app = FastAPI()

origins = [
    "http://localhost:3000",          
    "https://your-nextjs-domain.com", 
    "http://localhost:8000",          
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,              
    allow_credentials=True,             
    allow_methods=["*"],               
    allow_headers=["*"],                
)

with open('app/models/bag_model.pkl', 'rb') as file:
    model = pickle.load(file)

disease_model = load_model('app/models/wheat_leaf_cnn_model.h5')

crop_list = [
    'rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas', 'mothbeans',
    'mungbean', 'blackgram', 'lentil', 'pomegranate', 'banana', 'mango',
    'grapes', 'watermelon', 'muskmelon', 'apple', 'orange', 'papaya',
    'coconut', 'cotton', 'jute', 'coffee'
]

disease_classes = ['Healthy' 'Rust', 'Blight']

class PredictionInput(BaseModel):
    nitrogen: float
    phosphorus: float
    potassium: float
    temp: float
    humidity: float
    ph: float
    rainfall: float

class Disease(BaseModel):
    disease_id: int
    disease_name: str
    crop_affected: List[str]
    symptoms: Dict[str, List[str]]
    causes: List[str]
    severity: str
    geographical_distribution: List[str]
    treatment_options: List[str]
    prevention_tips: List[str]

def prepare_image(image: Image.Image):
    image = image.resize((128, 128)) 
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)  
    img_array = img_array / 255.0  
    return img_array

@app.post('/predict-crop')
def predict(input: PredictionInput):
    input_data = np.array([[input.nitrogen, input.phosphorus, input.potassium, 
                            input.temp, input.humidity, input.ph, input.rainfall]])
    
    prediction = model.predict(input_data)
    predicted_index = prediction[0].item()
    
    if predicted_index >= len(crop_list):
        return {"error": "Prediction index out of range"}
    
    predicted_crop = crop_list[predicted_index]
    
    return {'prediction': predicted_crop}

@app.get("/diseases", response_model=List[Disease])
def get_all_diseases():
    return disease_data

@app.get("/diseases/{name}", response_model=Disease)
def get_disease_by_name(name: str):
    for disease in disease_data:
        if disease["disease_name"].lower() == name.lower():
            return disease
    return {"error": "Disease not found"}

@app.get("/search", response_model=List[Disease])
def search_diseases(crop: Optional[str] = None, symptom: Optional[str] = None):
    result = []
    for disease in disease_data:
        if crop and crop.lower() in [c.lower() for c in disease["crop_affected"]]:
            result.append(disease)
        elif symptom:
            primary_symptoms = [s.lower() for s in disease["symptoms"]["primary"]]
            secondary_symptoms = [s.lower() for s in disease["symptoms"]["secondary"]]
            if symptom.lower() in primary_symptoms or symptom.lower() in secondary_symptoms:
                result.append(disease)
    return result


@app.post("/predict-leaf-disease/")
async def predict(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        img_array = prepare_image(image)
        predictions = model.predict(img_array)
        predicted_class = disease_classes[np.argmax(predictions)]
        
        return {"prediction": predicted_class}
    
    except Exception as e:
        return {"error": str(e)}