import dlib
import cv2
import pickle
import os
import numpy as np

def face_detection(image):
    face_detector = dlib.cnn_face_detection_model_v1('algorithm/mmod_human_face_detector.dat')
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # detect faces in the grayscale image
    rects = face_detector(image_rgb, 1)
    # return the bounding boxes
    return list(map(lambda x:x.rect, rects))

def face_encodings(image):
    # compute the facial embeddings for each face 
    # in the input image. the `compute_face_descriptor` 
    # function returns a 128-d vector that describes the face in an image
    shape_predictor = dlib.shape_predictor("algorithm/shape_predictor_68_face_landmarks.dat")
    face_encoder = dlib.face_recognition_model_v1("algorithm/dlib_face_recognition_resnet_model_v1.dat")
    face_box = face_detection(image)[0] #Assume only 1 face
    shape = shape_predictor(image, face_box)
    face_chip = dlib.get_face_chip(image, shape)   
    return face_encoder.compute_face_descriptor(face_chip)

def nb_of_matches(known_encodings, unknown_encoding):
    # compute the Euclidean distance between the current face encoding 
    # and all the face encodings in the database
    distances = np.linalg.norm(np.array(known_encodings) - unknown_encoding, axis=1)
    # keep only the distances that are less than the threshold
    small_distances = distances <= 0.6
    # return the number of matches
    return sum(small_distances)

def face_predict(image):
    with open("algorithm/encodings.pickle", "rb") as f:
        name_encodings_dict = pickle.load(f)
    questioned_face = face_encodings(image)
    counts = {}

    for (name, encodings) in name_encodings_dict.items():
        # compute the number of matches between the current encoding and the encodings 
        # of the known faces and store the number of matches in the dictionary
        counts[name] = nb_of_matches(encodings, questioned_face)
    # check if all the number of matches are equal to 0
    # if there is no match for any name, then we set the name to "Unknown"
    if all(count == 0 for count in counts.values()):
        name = "Unknown"
    # otherwise, we get the name with the highest number of matches
    else:
        name = max(counts, key=counts.get)
    return name

def add_face_to_model(name, image_files):
    with open("algorithm/encodings.pickle", "rb") as f:
        name_encodings_dict = pickle.load(f)

    nb_current_image = 1
    for image_file in image_files:
        print(f"Image processed {nb_current_image}/{len(image_files)} for {name}")
        npimg = np.frombuffer(image_file.file.read(), np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        encodings = face_encodings(image)
        e = name_encodings_dict.get(name, [])
        e.append(encodings)
        name_encodings_dict[name] = e
        nb_current_image += 1

    with open("algorithm/encodings.pickle", "wb") as f:
        pickle.dump(name_encodings_dict, f)

def model_info():
    with open("algorithm/encodings.pickle", "rb") as f:
        name_encodings_dict = pickle.load(f)
    return {k:len(v) for k,v in name_encodings_dict.items()}

def prepare_training_data(data_folder_path):
    dirs = os.listdir(data_folder_path)
    name_encodings_dict = {}
    for dir_name in dirs:
        nb_current_image = 1
        subject_dir_path = data_folder_path + "/" + dir_name
        subject_images_names = os.listdir(subject_dir_path)

        for image_name in subject_images_names:
            if image_name.startswith("."):
                continue;
            image_path = subject_dir_path + "/" + image_name
            print(f"Image processed {nb_current_image}/{len(subject_images_names)} for {dir_name}")
            image = cv2.imread(image_path)
            encodings = face_encodings(image)
            name = dir_name
            e = name_encodings_dict.get(name, [])
            e.append(encodings)
            name_encodings_dict[name] = e
            nb_current_image += 1

    with open("algorithm/encodings.pickle", "wb") as f:
        pickle.dump(name_encodings_dict, f)