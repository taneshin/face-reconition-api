import cv2
import numpy as np
from typing import List
from fastapi import APIRouter, UploadFile
from starlette.status import HTTP_201_CREATED
from algorithm.utils import face_predict, add_face_to_model, model_info

router = APIRouter()

@router.post('/predict')
async def upload_face(file: UploadFile):
    npimg = np.frombuffer(await file.read(), np.uint8)
    identity = face_predict(cv2.imdecode(npimg, cv2.IMREAD_COLOR))
    return {"identity": identity}

@router.post('/add', status_code=HTTP_201_CREATED)
async def add_new_person(name: str, files: List[UploadFile]):
    add_face_to_model(name, files)
    return {"name": name}

@router.get('/model')
async def retrieve_model_info():
    return model_info()