import cv2
import numpy as np
from typing import List
from fastapi import APIRouter, UploadFile, HTTPException, status
from starlette.status import HTTP_201_CREATED, HTTP_204_NO_CONTENT
from algorithm.model import face_predict, add_face_to_model
from algorithm.utils import model_info, delete_pickle

router = APIRouter()

@router.get('/model')
async def retrieve_model_info():
    if model_info():
        return model_info()
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model pickle file does not exist",
        )

@router.delete("/model", status_code=HTTP_204_NO_CONTENT)
async def delete_model():
    if delete_pickle():
        return {"msg": "Successfully deleted."}
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model pickle file does not exist",
        )

@router.post('/predict')
async def predict_face(file: UploadFile):
    npimg = np.frombuffer(await file.read(), np.uint8)
    identity, distance = face_predict(cv2.imdecode(npimg, cv2.IMREAD_COLOR))
    if identity:
        return {"identity": identity,
                "euclidean distance": round(distance, 2)}
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model pickle file does not exist",
        )

@router.post('/add', status_code=HTTP_201_CREATED)
async def add_new_person(name: str, files: List[UploadFile]):
    add_face_to_model(name, files)
    return {"name": name}