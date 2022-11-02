import cv2
import numpy as np
from fastapi import APIRouter
from fastapi import File
from fastapi import UploadFile
from algorithm.recognition import predict, create_recognizer

router = APIRouter()
face_recognizer = create_recognizer()

def load_image_into_numpy_array(data):
    return np.array(Image.open(BytesIO(data)))

@router.post('/upload')
async def upload_face(file: UploadFile = File(...)):
    npimg = np.frombuffer(await file.read(), np.uint8)
    identity, loss = predict(face_recognizer, cv2.imdecode(npimg, cv2.IMREAD_COLOR))
    return {"identity": identity, "loss": loss}
