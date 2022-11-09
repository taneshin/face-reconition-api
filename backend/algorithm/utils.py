import os
import pickle
from fastapi import HTTPException, status

def read_pickle():
    try:
        with open("algorithm/encodings.pickle", "rb") as f:
            name_encodings_dict = pickle.load(f)
        return name_encodings_dict
    except:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model pickle file does not exist",
        )


def model_info():
    return {k:len(v) for k,v in read_pickle().items()}

def delete_pickle():
    try:
        os.remove('algorithm/encodings.pickle')
    except:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model pickle file does not exist",
        )