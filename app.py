import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os
import logging
import warnings
import io
from fastapi import FastAPI, HTTPException, status, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

class ResponseModel(BaseModel):
    is_grape_leaf: bool
    grape_probability: float
    message: str

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore', category=FutureWarning)
tf.get_logger().setLevel('ERROR')

class GrapeDiseasePredictor:
    def __init__(self, classifier_model_path):
        """
        Inicializar el predictor
        """
        try:
            self.classifier_model = tf.keras.models.load_model(classifier_model_path)
            logger.info(f"Modelo clasificador cargado exitosamente desde {classifier_model_path}")
            
        except Exception as e:
            logger.error(f"Error cargando los modelos: {e}")
            raise
            
        self.IMG_SIZE = (128, 128)
    
        # Asumiendo que el modelo retorna [No es hoja de uva, Es hoja de uva]
        self.grape_leaf_threshold = 0.5  # Umbral para considerar que es hoja de uva
        
    def preprocess_image(self, image):
        """
        Preprocesar la imagen para los modelos
        """
        try:
            # Convertir PIL a numpy array si es necesario
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            # Verificar que la imagen tiene el formato correcto
            if len(image.shape) != 3 or image.shape[2] != 3:
                raise ValueError("La imagen debe ser en formato RGB")
            
            # Redimensionar a 128x128
            image_resized = cv2.resize(image, self.IMG_SIZE)
            
            # Normalizar (0-1)
            image_normalized = image_resized.astype(np.float32) / 255.0
            
            # Añadir dimensión de batch
            image_batch = np.expand_dims(image_normalized, axis=0)
            
            return image_batch, image_resized, image_normalized
            
        except Exception as e:
            logger.error(f"Error en preprocesamiento: {e}")
            raise
    
    def is_grape_leaf(self, image):
        """
        Verificar si la imagen es una hoja de uva usando el modelo clasificador
        """
        try:
            # Preprocesar imagen
            image_batch, _, _ = self.preprocess_image(image)
            
            # Hacer predicción con el modelo clasificador
            prediction = self.classifier_model.predict(image_batch, verbose=0, batch_size=1)
            
            # Obtener probabilidad de que sea hoja de uva
            # Ajustar este índice según cómo esté configurado tu modelo
            # Si el modelo devuelve [prob_no_hoja, prob_hoja], usar índice 1
            # Si devuelve solo una probabilidad, usar índice 0
            if prediction.shape[1] == 2:
                grape_leaf_probability = float(prediction[0][1])  # Probabilidad de ser hoja de uva
            else:
                grape_leaf_probability = float(prediction[0][0])  # Si es un solo valor
            
            is_grape = grape_leaf_probability < self.grape_leaf_threshold
            
            return is_grape, grape_leaf_probability
            
        except Exception as e:
            logger.error(f"Error verificando si es hoja de uva: {e}")
            raise
    
    def predict(self, image):
        """
        Realizar predicción completa
        """
        try:
            is_grape, grape_probability = self.is_grape_leaf(image)

            message = "✅ la imagen parece ser una hoja de uva" if is_grape else f"❌ La imagen no parece ser una hoja de uva (confianza: {grape_probability:.1%})"
            data = {
                'is_grape_leaf': is_grape,
                'grape_probability': grape_probability,
                'message': message
            }
            return ResponseModel(**data)
            
        except Exception as e:
            logger.error(f"Error en predicción completa: {e}")
            raise

app = FastAPI(title="Grape Leaf verification API", 
              version="1.0", 
              description="API para verificar si una imagen es una hoja de uva")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

classifier_model_path="grape_classifier.keras"

predictor = GrapeDiseasePredictor(classifier_model_path)

@app.post("/predict")
async def predict_endpoint(image: UploadFile = File(...)):
    if image.filename is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No se proporcionó un archivo")
    ext = image.filename.split('.')[-1].lower()
    if ext not in ['jpg', 'jpeg']:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="El archivo debe ser una imagen jpg o jpeg")
    try:
        image.file.seek(0)
        image_file = image.file.read()
        image_pil = Image.open(io.BytesIO(image_file))
        prediction = predictor.predict(image_pil)
        return prediction
    except Exception as e:
        logger.error(f"Error en la solicitud POST: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error en la solicitud")
