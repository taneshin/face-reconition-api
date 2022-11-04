import os
import pickle

def read_pickle():
    try:
        with open("algorithm/encodings.pickle", "rb") as f:
            name_encodings_dict = pickle.load(f)
        return name_encodings_dict
    except Exception as e:
        return False

def model_info():
    if read_pickle():
        return {k:len(v) for k,v in read_pickle().items()}
    else:
        return False

def delete_pickle():
    try:
        os.remove('algorithm/encodings.pickle')
        return True
    except Exception:
        return False