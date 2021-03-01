#!/usr/bin/python


import numpy as np
import matplotlib.image as mpimg
import os
import cv2
# It requires the pyERA library
from pyERA.som import Som
from pyERA.utils import ExponentialDecay


def loadImages(path):
    # return array of images

    imagesList = os.listdir(path)
    loadedImages = []
    for image in np.sort(imagesList):
        img = mpimg.imread(path + image)
        resize_img = cv2.resize(img, (32, 32))
        resize_img_gs = cv2.cvtColor(resize_img, cv2.COLOR_BGR2GRAY)
        # TODO: Ask about image foveation
        loadedImages.append(resize_img_gs)

    return loadedImages


def main():
    # Define output directory to save sensory som
    output_path = "./output/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Load the data from the sensory modality
    path = "./data/pepper_images/"
    # images in an array
    data = loadImages(path)
    input_size = data[0].flatten().shape[0]

    # Init the SOM
    som_size = 64
    my_som = Som(
        matrix_size=som_size,
        input_size=input_size,
        low=np.min(data),
        high=np.max(data),
        round_values=True
    )

    tot_epoch = len(data)

    my_learning_rate = ExponentialDecay(
        starter_value=0.5,
        decay_step=int(tot_epoch/10),
        decay_rate=0.90,
        staircase=True
    )
    my_radius = ExponentialDecay(
        starter_value=np.rint(som_size/3),
        decay_step=int(tot_epoch/12),
        decay_rate=0.90,
        staircase=True
    )

    # Starting the Learning
    for epoch in range(1, tot_epoch):
        # Updating the learning rate and the radius
        learning_rate = my_learning_rate.return_decayed_value(
            global_step=epoch
        )
        radius = my_radius.return_decayed_value(global_step=epoch)

        input_vector = data[epoch - 1].flatten()
        # print(input_vector)

        # Estimating the BMU coordinates
        bmu_index = my_som.return_BMU_index(input_vector)
        bmu_weights = my_som.get_unit_weights(
            bmu_index[0], bmu_index[1])

        # Getting the BMU neighborhood
        bmu_neighborhood_list = my_som.return_unit_round_neighborhood(
            bmu_index[0], bmu_index[1], radius=radius
        )

        # Learning step
        my_som.training_single_step(
            input_vector,
            units_list=bmu_neighborhood_list,
            learning_rate=learning_rate,
            radius=radius,
            weighted_distance=False
        )

        print("")
        print("Epoch: " + str(epoch))
        print("Learning Rate: " + str(learning_rate))
        print("Radius: " + str(radius))
        print("Input vector: " + str(input_vector))
        print("BMU index: " + str(bmu_index))
        print("BMU weights: " + str(bmu_weights))
        # print("BMU NEIGHBORHOOD: " + str(bmu_neighborhood_list))

    # Saving the network
    file_name = output_path + "som_sensory.npz"
    print("Saving the network in: " + str(file_name))
    my_som.save(path=output_path, name="som_sensory")


if __name__ == "__main__":
    main()
