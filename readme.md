# Repository for basic facial recognition API

## Setup

1. Open up your console and clone repository locally.
2. Install the library dependencies.
```sh
pip3 install -r requirements.txt
```
3. Launch and test API via [localhost:8000/docs](localhost:8000/docs)
```sh
uvicorn main:app
```

## API Routes

### GET
- /model - retrieve information about the model
<br>JSON response with pairs of identity_name:number of trained photos


### DELETE

- /model - deletes the pickle file containing the recognised faces

### POST

- /predict - predicts the idenity of a face with preset euclidean distance of 0.6
<br>JSON response with predicted idenity and euclidean distance. If greater than 0.6, identity will be "unknown"

- /authenticate - Takes in a name and optional threshold parameter and confirms the identity of a face
<br>Will return either true or false

- /add - Adds faces to the system. 
<br>JSON response with the filenames of successfully added and rejected images

## Models used:
1. Facial detection - [YuNet](https://medium.com/@silkworm/yunet-ultra-high-performance-face-detection-in-opencv-a-good-solution-for-real-time-poc-b01063e251d5)
2. Facial alignment - [Dlib's 68 point face landmark predictor](http://dlib.net/face_landmark_detection.py.html)
3. Facial encoding - [Dlib's ResNet-34 inspired model](https://paperswithcode.com/paper/dlib-ml-a-machine-learning-toolkit)

## How it works:

1. Faces are first detected using the Yunet detector.
2. Faces are then aligned and cropped by using the landmark predictor. This will keep faces consistent for better results
3. Faces are then converted to 128 dimension vector encodings using Dlib's ResNet-34 recogniser.
4. Faces can then be recognised by comparing thier 128-d vectors with each other. Images that have similar vectors (i.e. low euclidean distance) are considered to be of the same face.
