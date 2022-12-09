from fastapi import HTTPException, status

def face_detection_yunet(image):
    import cv2
    import dlib
    face_detector = cv2.FaceDetectorYN.create("algorithm/face_detection_yunet_2022mar.onnx", "", (0,0))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image_rgb.shape
    face_detector.setInputSize((w,h))
    _, faces = face_detector.detect(image_rgb)
    try:
        if len(faces) > 1:
            raise HTTPException(
                status_code=status.HTTP_406_NOT_ACCEPTABLE,
                detail="Image contains multiple faces",
            )
        else:
            face = faces[0]
            # https://github.com/keyurr2/facial-landmarks/blob/master/facial_landmarks.py#L88
            return dlib.rectangle(face[0], int(face[1] * 1.15), int((face[0]+face[2]) * 1.05), face[1]+face[3])
    except TypeError:
        raise HTTPException(
            status_code=status.HTTP_406_NOT_ACCEPTABLE,
            detail="Image contains no faces",
        )
    
def face_encodings(image):
    import dlib
    # compute the facial embeddings for each face 
    # in the input image. the `compute_face_descriptor` 
    # function returns a 128-d vector that describes the face in an image
    shape_predictor = dlib.shape_predictor("algorithm/shape_predictor_68_face_landmarks.dat")
    face_encoder = dlib.face_recognition_model_v1("algorithm/dlib_face_recognition_resnet_model_v1.dat")
    face_box = face_detection_yunet(image)
    shape = shape_predictor(image, face_box)
    face_chip = dlib.get_face_chip(image, shape)   
    return face_encoder.compute_face_descriptor(face_chip)