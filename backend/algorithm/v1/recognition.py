import cv2
from .detection import detect_face

def prepare_training_data(data_folder_path):
    dirs = os.listdir(data_folder_path)
    faces = []
    labels = []
    reject = 0
    
    for dir_name in dirs:
        label = int(dir_name)
        if label > 5:
            continue
        subject_dir_path = data_folder_path + "/" + dir_name
        subject_images_names = os.listdir(subject_dir_path)

        for image_name in subject_images_names:
            if image_name.startswith("."):
                continue;
            image_path = subject_dir_path + "/" + image_name
            image = cv2.imread(image_path)
            face, rect = detect_face(image)

            if face is not None:
                faces.append(face)
                labels.append(label)
            else:
                reject+=1
    print(f'rejected {reject}')
    return faces, labels

def create_recognizer():
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    # face_recognizer.train(faces, np.array(labels))
    face_recognizer.read("algorithm/model_celeb.yml")
    return face_recognizer

def predict(face_recognizer, test_img):
    img = test_img.copy()
    face, rect = detect_face(img)
    return face_recognizer.predict(face)