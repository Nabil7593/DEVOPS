from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import requests
import os
from dotenv import load_dotenv

# 🌍 Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# 🚀 Config FastAPI
app = FastAPI()

# 🔐 Endpoint et clé Azure ML depuis .env
endpoint = os.getenv("AZUREML_ENDPOINT")
api_key = os.getenv("AZUREML_API_KEY")

# Headers pour requête POST vers Azure ML
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

# 📥 Schéma d’entrée
class InputData(BaseModel):
    data: List[List[float]]  # Liste de lignes de features (ex: [[5.1, 3.5, 1.4, 0.2]])

# 🏠 Page d'accueil test
@app.get("/")
def home():
    return {"message": "API IRIS FastAPI connectée à Azure ML"}

# 🔮 Endpoint /predict
@app.post("/predict")
def predict(input_data: InputData):
    payload = {"data": input_data.data}
    response = requests.post(endpoint, headers=headers, json=payload)

    if response.status_code != 200:
        return {"error": response.text}

    return {"prediction": response.json()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
