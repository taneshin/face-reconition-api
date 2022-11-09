import cv2
import numpy as np
from typing import List
from fastapi import APIRouter, UploadFile
from starlette.status import HTTP_201_CREATED, HTTP_204_NO_CONTENT
from algorithm.model import face_predict, add_face_to_model
from algorithm.utils import model_info, delete_pickle

router = APIRouter()

@router.get('/model')
async def retrieve_model_info():
    if model_info():
        return model_info()

@router.delete("/model", status_code=HTTP_204_NO_CONTENT)
async def delete_model():
    try:
        delete_pickle()
        return {"msg": "Successfully deleted."}
    except Exception as e:
        raise e

@router.post('/predict')
async def predict_face(file: UploadFile):
    npimg = np.frombuffer(await file.read(), np.uint8)
    try:
        identity, distance = face_predict(cv2.imdecode(npimg, cv2.IMREAD_COLOR))
    except Exception as e:
        raise e
    else:
        return {"identity": identity,
                "euclidean distance": round(distance, 2)}

@router.post('/authenticate')
async def authenticate(name: str, file: UploadFile, threshold: float = 0.6):
    npimg = np.frombuffer(await file.read(), np.uint8)
    try:
        identity, _ = face_predict(cv2.imdecode(npimg, cv2.IMREAD_COLOR), threshold)
    except Exception as e:
        raise e
    else:
        return {"check": identity == name}

@router.post('/add', status_code=HTTP_201_CREATED)
async def add_new_person(name: str, files: List[UploadFile]):
    add_face_to_model(name, files)
    return {"added": name}