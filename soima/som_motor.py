#!/usr/bin/python


import numpy as np
import matplotlib.pyplot as plt
import os
# It requires the pyERA library
from pyERA.som import Som
from pyERA.utils import ExponentialDecay


def save_map_image(
    save_path, size, weight_matrix, yaw_max_range=90.0, pitch_max_range=90.0
):
    # x = np.arange(0, size, 1) + 0.5
    # y = np.arange(0, size, 1) + 0.5

    fig = plt.figure()
    # plt.title('iCub Head Pose SOM')
    ax = fig.gca()
    ax.set_xlim([0, size])
    ax.set_ylim([0, size])
    ax.set_xticks(np.arange(1, size+1, 1))
    ax.set_yticks(np.arange(1, size+1, 1))

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ticklines = ax.get_xticklines() + ax.get_yticklines()
    gridlines = ax.get_xgridlines() + ax.get_ygridlines()
    # ticklabels = ax.get_xticklabels() + ax.get_yticklabels()

    for line in ticklines:
        line.set_linewidth(2)

    for line in gridlines:
        line.set_linestyle('-')
        line.set_color("grey")

    tot_rows = weight_matrix.shape[0]
    tot_cols = weight_matrix.shape[1]

    for row in range(0, tot_rows):
        for col in range(0, tot_cols):
            yaw = weight_matrix[row, col, 0]
            pitch = weight_matrix[row, col, 1]
            # if(pitch > 30.0 or pitch < -30): pitch_max_range=90.0
            # else: pitch_max_range = 30.0
            yaw_arrow = (yaw / yaw_max_range) * 0.4
            pitch_arrow = (pitch / pitch_max_range) * 0.4
            ax.arrow(
                row+0.5,
                col+0.5,
                yaw_arrow,
                pitch_arrow,
                head_width=0.05,
                head_length=0.05,
                fc='k',
                ec='k'
            )

    # s is the dot area and c is the color
    # plt.scatter(x, y, s=3.0, c="black")
    # plt.grid()
    # plt.show()
    ax.axis('off')
    plt.savefig(save_path, dpi=300, facecolor='white')
    plt.close('all')


def main():
    # Set to True if you want to save the SOM images inside a folder.
    SAVE_IMAGE = True
    output_path = "./output/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Load the data
    data = np.loadtxt('./data/joint_positions.txt')
    tot_epoch = data.shape[0]
    data_degrees = data.copy() * 180 / np.pi
    data_degree_max = np.max(abs(data_degrees))
    normalized_data = (data - np.min(data)) / \
        (np.max(data) - np.min(data))

    # Init the SOM
    som_size = 16
    my_som = Som(
        matrix_size=som_size,
        input_size=2,
        low=np.min(normalized_data),
        high=np.max(normalized_data),
        round_values=True
    )

    my_learning_rate = ExponentialDecay(
        starter_value=0.5,
        decay_step=int(tot_epoch/5),
        decay_rate=0.9,
        staircase=True
    )
    my_radius = ExponentialDecay(
        starter_value=np.rint(som_size/3),
        decay_step=int(tot_epoch/6),
        decay_rate=0.90,
        staircase=True
    )

    # Starting the Learning
    for epoch in range(1, tot_epoch):

        # Saving the image associated with the SOM weights
        if (SAVE_IMAGE is True) and (epoch % 100 == 0):
            save_path = output_path + str(epoch) + ".jpg"
            weight_matrix_degrees = (my_som.return_weights_matrix() * \
                (np.max(data) - np.min(data)) + np.min(data)) * 180 / np.pi
            save_map_image(
                save_path,
                som_size,
                weight_matrix_degrees,
                data_degree_max,
                data_degree_max
            )

        # Updating the learning rate and the radius
        learning_rate = my_learning_rate.return_decayed_value(
            global_step=epoch
        )
        radius = my_radius.return_decayed_value(global_step=epoch)

        # Load input vector
        input_vector = normalized_data[epoch - 1]

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
    file_name = output_path + "som_motor.npz"
    print("Saving the network in: " + str(file_name))
    my_som.save(path=output_path, name="som_motor")

    # img = np.rint(my_som.return_weights_matrix())
    # plt.axis("off")
    # plt.imshow(img)
    # plt.show()


if __name__ == "__main__":
    main()
