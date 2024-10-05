from utils.dataset import load_dataset
from utils.camera_utils import Camera
from munch import munchify
import yaml
import sys
import torch
import numpy as np
from gaussian_splatting.scene.gaussian_model import GaussianModel
from utils.config_utils import load_config
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
import pydbow3 as bow
import cv2
from monogs_ros.utils.orb_extractor import ORBExtractor


def load_image_descriptors(dataset):
    descriptor_list = []
    print("Processing images in dataset....")
    for idx in range(len(dataset)):
        # Load images and convert them to grayscale
        img_color, _, _ = dataset[idx]  # Load first image in grayscale

        _, descriptors = calc_keypoint_descriptor_from_dataset_image(img_color)

        descriptor_list.append(descriptors)
    print("Done!!!")

    return descriptor_list

def calc_keypoint_descriptor_from_dataset_image(tensor_img_color):
        numpy_img = tensor_img_color.cpu().permute(1, 2, 0).numpy()
        numpy_img = (numpy_img * 255).astype(np.uint8)

        img_gray = cv2.cvtColor(numpy_img, cv2.COLOR_RGB2GRAY)

        # Initialize ORB feature detector
        orb = cv2.ORB_create()

        # Detect keypoints and descriptors
        keypoints, descriptors = orb.detectAndCompute(img_gray, None)
        descriptors = np.array(descriptors, dtype=np.float32)


        return keypoints, descriptors

def test_vocab_creation(dataset):
    # Need to add dbow3 here
    voc = bow.Vocabulary()
    descriptor_list = []

    print("Processing images in dataset....")
    for idx in range(len(dataset)):
        if idx > 20:
            break
        # Load images and convert them to grayscale
        img_color, _, _ = dataset[idx]  # Load first image in grayscale

        _, descriptors = calc_keypoint_descriptor_from_dataset_image(img_color)

        descriptor_list.append(descriptors)
    print("Done!!!")

    # Create a vocabulary from the descriptors (using the first image as an example)
    vocab = bow.Vocabulary()
    print("Creating vocabulary....")
    vocab.create(descriptor_list)
    print("Done!!!")


    for idx1 in range(len(dataset)):
        for idx2 in range(len(dataset)):

            # Load images and convert them to grayscale
            img1_color, _, _ = dataset[idx1]  # Load first image in grayscale
            _, descriptors1 = calc_keypoint_descriptor_from_dataset_image(img1_color)


            # Load images and convert them to grayscale
            img2_color, _, _ = dataset[idx2]  # Load first image in grayscale
            _, descriptors2 = calc_keypoint_descriptor_from_dataset_image(img2_color)

            # Transform descriptors into bag-of-words vectors
            bow_vector1 = vocab.transform(descriptors1)
            bow_vector2 = vocab.transform(descriptors2)

            # Compute similarity between images
            similarity_score = vocab.score(bow_vector1, bow_vector2)

            print(f'Similarity score between image1 and image2: {similarity_score}')

def test_database(dataset):
    voc = bow.Vocabulary("orbvoc.dbow3")
    db = bow.Database()
    db.setVocabulary(voc, False, 0)

    descriptor_list = load_image_descriptors(dataset)

    for i, desc in enumerate(descriptor_list):
        if i > 20:
            break
        db.add(desc)


    for i, desc in enumerate(descriptor_list):
        #print(i)
        results = db.query(desc, 4)
        #print(results[0].Score)

    print("Saving DB...")
    db.save("database")
    print("Done!!!")

    print("Loading DB....")
    db_new = bow.Database("database")
    print("Done!!!")


def place_recognition_test(dataset):

    voc = bow.Vocabulary("orbvoc.dbow3", 10, 5, bow.WeightingType.TF_IDF, bow.ScoringType.DOT_PRODUCT)
    db = bow.Database()
    db.setVocabulary(voc, False, 0)

    descriptor_list = load_image_descriptors(dataset)
    db_entry_dict = {}
    db_count = 0
    for i, desc in enumerate(descriptor_list):
        db.add(desc)
        db_entry_dict[db_count] = i
        db_count += 1

    idx = 210
    num_results = 100
    desc = descriptor_list[idx]
    #print(i)
    results = db.query(desc, num_results)
    for i in range(num_results):
        img_color1, _, _ = dataset[idx]
        keypoints1, descriptors1 = calc_keypoint_descriptor_from_dataset_image(img_color1)

        numpy_image1 = img_color1.permute(1, 2, 0).cpu().numpy()
        numpy_image1 = (numpy_image1 * 255).astype(np.uint8)
        opencv_image1 = cv2.cvtColor(numpy_image1, cv2.COLOR_RGB2BGR)

        # Draw keypoints on the images
        img1_keypoints = cv2.drawKeypoints(opencv_image1, keypoints1, None, color=(0,255,0), flags=0)


        q_id = db_entry_dict[results[i].Id]
        img_color2, _, _ = dataset[q_id]
        keypoints2, descriptors2 = calc_keypoint_descriptor_from_dataset_image(img_color2)

        # Transform descriptors into bag-of-words vectors
        bow_vector1 = voc.transform(descriptors1)
        bow_vector2 = voc.transform(descriptors2)

        # Compute similarity between images
        similarity_score = voc.score(bow_vector1, bow_vector2)
        print(f"For {i}-th result, retrieved entry #{results[i].Id} from db and image id:{db_entry_dict[results[i].Id]} with score:{results[i].Score}, similarity score: {similarity_score} and num words in common : {results[i].nWords}")

        numpy_image2 = img_color2.permute(1, 2, 0).cpu().numpy()
        numpy_image2 = (numpy_image2 * 255).astype(np.uint8)
        opencv_image2 = cv2.cvtColor(numpy_image2, cv2.COLOR_RGB2BGR)

        # Draw keypoints on the images
        img2_keypoints = cv2.drawKeypoints(opencv_image2, keypoints2, None, color=(0,255,0), flags=0)


        # Concatenate the two images horizontally
        concatenated_images = np.hstack((img1_keypoints, img2_keypoints))

        # # Display the result
        # cv2.imshow('Keypoints on Images', concatenated_images)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


def main():
    config_file = "configs/rgbd/tum/fr1_desk.yaml"
    with open(config_file, "r") as yml:
        config = yaml.safe_load(yml)

    config = load_config(config_file)
    model_params = munchify(config["model_params"])
    dataset = load_dataset(model_params, model_params.source_path, config=config)
    idx = 3
    projection_matrix = getProjectionMatrix2(
        znear=0.01,
        zfar=100.0,
        fx=dataset.fx,
        fy=dataset.fy,
        cx=dataset.cx,
        cy=dataset.cy,
        W=dataset.width,
        H=dataset.height,
    ).transpose(0, 1)
    projection_matrix = projection_matrix.to(device="cuda:0")
    viewpoint = Camera.init_from_dataset(dataset, idx, projection_matrix)
    viewpoint.ORBExtract()
    print(len(viewpoint.keypoints))
    print(viewpoint.descriptors.shape)

    image_np = viewpoint.original_image.permute(1,2,0).cpu().numpy()
    image_np = (image_np*255).astype(np.uint8)
    img_gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

    image_with_keypoints = cv2.drawKeypoints(img_gray, viewpoint.keypoints, None, color=(0, 255, 0), flags=cv2.DrawMatchesFlags_DEFAULT)

    cv2.imshow('Keypoints', image_with_keypoints)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #test_vocab_creation(dataset)

    #place_recognition_test(dataset)



    # Need to check place recognition



if __name__ == "__main__":
    main()
