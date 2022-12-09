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
    return model_info()

@router.delete("/model", status_code=HTTP_204_NO_CONTENT)
async def delete_model():
    delete_pickle()
    return {"msg": "Successfully deleted."}

@router.post('/find')
async def predict_face(file: UploadFile):
    npimg = np.frombuffer(await file.read(), np.uint8)
    identity, distance = face_predict(cv2.imdecode(npimg, cv2.IMREAD_COLOR))
    return {"identity": identity
            , "distance": distance
            , "similarity_metric": "euclidean"
            }

@router.post('/verify ')
async def authenticate(name: str, file: UploadFile, max_threshold_to_verify: float = 0.6):
    npimg = np.frombuffer(await file.read(), np.uint8)
    identity, distance = face_predict(cv2.imdecode(npimg, cv2.IMREAD_COLOR), threshold)

    return {"verified": identity == name
            , "score": score
            , "distance": distance
            }

@router.post('/add', status_code=HTTP_201_CREATED)
async def add_new_person(name: str, files: List[UploadFile]):
    added, rejected = add_face_to_model(name, files)
    return {"added": added
            , "rejected": rejected
            }