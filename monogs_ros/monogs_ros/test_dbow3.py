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



def load_image_descriptors(dataset):
    descriptor_list = []
    print("Processing images in dataset....")
    for idx in range(len(dataset)):
        # Load images and convert them to grayscale
        img_color, _, _ = dataset[idx]  # Load first image in grayscale

        descriptors = calc_descriptor_from_dataset_image(img_color)

        descriptor_list.append(descriptors)
    print("Done!!!")

    return descriptor_list

def calc_descriptor_from_dataset_image(tensor_img_color):
        numpy_img = tensor_img_color.cpu().permute(1, 2, 0).numpy()
        numpy_img = (numpy_img * 255).astype(np.uint8)

        img_gray = cv2.cvtColor(numpy_img, cv2.COLOR_RGB2GRAY)

        # Initialize ORB feature detector
        orb = cv2.ORB_create()

        # Detect keypoints and descriptors
        keypoints, descriptors = orb.detectAndCompute(img_gray, None)
        descriptors = np.array(descriptors, dtype=np.float32)

        return descriptors

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

        descriptors = calc_descriptor_from_dataset_image(img_color)

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
            descriptors1 = calc_descriptor_from_dataset_image(img1_color)


            # Load images and convert them to grayscale
            img2_color, _, _ = dataset[idx2]  # Load first image in grayscale
            descriptors2 = calc_descriptor_from_dataset_image(img2_color)

            # Transform descriptors into bag-of-words vectors
            bow_vector1 = vocab.transform(descriptors1)
            bow_vector2 = vocab.transform(descriptors2)

            # Compute similarity between images
            similarity_score = vocab.score(bow_vector1, bow_vector2)

            print(f'Similarity score between image1 and image2: {similarity_score}')

def test_database(dataset):
    voc = bow.Vocabulary("ORBvoc.txt")
    db = bow.Database()
    db.setVocabulary(voc, False, 0)

    descriptor_list = load_image_descriptors(dataset)

    for i, desc in enumerate(descriptor_list):
        if i > 20:
            break
        db.add(desc)


    for i, desc in enumerate(descriptor_list):
        print(i)
        results = db.query(desc, 4)
        print(results[0].Score)
        input('Press any key to continue')

    # print("Saving DB...")
    # db.save("test_db.db")
    # print("Done!!!")

    # print("Loading DB....")
    # db_new = bow.Database("test_db.db")
    # print("Done!!!")




def main():
    config_file = "configs/rgbd/tum/fr1_desk.yaml"
    with open(config_file, "r") as yml:
        config = yaml.safe_load(yml)

    config = load_config(config_file)
    model_params = munchify(config["model_params"])
    dataset = load_dataset(model_params, model_params.source_path, config=config)
    idx = 3
    gt_color, gt_depth, gt_pose = dataset[idx]

    #test_vocab_creation(dataset)

    test_database(dataset)




    # Need to check place recognition



if __name__ == "__main__":
    main()