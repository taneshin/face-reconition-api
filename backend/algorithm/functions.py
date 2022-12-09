def nb_of_matches(known_encodings, unknown_encoding, threshold):
    import numpy as np
    # compute the Euclidean distance between the current face encoding 
    # and all the face encodings in the database
    distances = np.linalg.norm(np.array(known_encodings) - unknown_encoding, axis=1)
    # keep only the distances that are less than the threshold
    small_distances = distances <= threshold
    # return the number of matches
    return sum(small_distances), np.average(distances)

def face_predict(image, threshold=0.6):
    from algorithm.utils import read_pickle
    name_encodings_dict = read_pickle()
    questioned_face = face_encodings(image)
    counts = {}

    for (name, encodings) in name_encodings_dict.items():
        # compute the number of matches between the current encoding and the encodings 
        # of the known faces and store the number of matches in the dictionary
        counts[name] = nb_of_matches(encodings, questioned_face, threshold)
    # check if all the number of matches are equal to 0
    # if there is no match for any name, then we set the name to "Unknown"
    if all(count[0] == 0 for count in counts.values()):
        name = "Unknown"
        e_dist = counts[min(counts, key=lambda x:counts[x][1])][1]
    # otherwise, we get the name with the highest number of matches
    else:
        name = max(counts, key=lambda x:counts[x][0])
        e_dist = counts[name][1]
    return name, e_dist

def add_face_to_model(name, image_files):
    from algorithm.utils import read_pickle
    import numpy as np
    import cv2
    import pickle
    import logging
    stats = [[],[]]
    try:
        name_encodings_dict = read_pickle()
    except:
        name_encodings_dict = {}

    nb_current_image = 1
    for image_file in image_files:
        print(f"Image processed {nb_current_image}/{len(image_files)} for {name}")
        npimg = np.frombuffer(image_file.file.read(), np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        try:
            encodings = face_encodings(image)
        except HTTPException as e:
            stats[1].append((image_file.filename,e.detail))
            logging.warning(f'{image_file.filename} has an error \"{e.detail}\"')
            continue
        else:
            stats[0].append(image_file.filename)
        e = name_encodings_dict.get(name, [])
        e.append(encodings)
        name_encodings_dict[name] = e
        nb_current_image += 1

    with open("algorithm/encodings.pickle", "wb") as f:
        pickle.dump(name_encodings_dict, f)
    
    return stats

def prepare_initial_bulk_pickle(data_folder_path):
    import cv2
    import pickle
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