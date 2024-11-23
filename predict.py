from flask import Flask, jsonify
import pickle
import numpy as np
from pydantic import BaseModel
from flask_pydantic import validate
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load the model from the pickle file
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

logger.info('Model loaded')

# Define the Pydantic model
class WineFeatures(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    chlorides: float
    free_sulfur_dioxide: int
    total_sulfur_dioxide: int
    density: float
    ph: float
    sulphates: float
    alcohol: float

    def to_numpy(self) -> np.ndarray:
        return np.array([[
            self.fixed_acidity,
            self.volatile_acidity,
            self.citric_acid,
            self.chlorides,
            self.free_sulfur_dioxide,
            self.total_sulfur_dioxide,
            self.density,
            self.ph,
            self.sulphates,
            self.alcohol
        ]])

@app.route('/predict', methods=['POST'])
@validate()
def predict(body: WineFeatures):
    logger.info('Predicting')
    prediction = model.predict(body.to_numpy())
    
    return jsonify({'quality': round(float(prediction[0]), 1)})


@app.route('/health', methods=['GET'])
def health():
    return 'OK'

if __name__ == '__main__':
    app.run()
