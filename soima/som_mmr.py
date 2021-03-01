import os
import numpy as np
from pyERA.som import Som
from pyERA.utils import ExponentialDecay
from som_sensory import loadImages


output_path = "./output/"
if not os.path.exists(output_path):
    os.makedirs(output_path)


# Load the data from different modalities
motor_data = np.loadtxt("./data/joint_positions.txt")
# Convert motor data from radians to degrees
motor_data_degrees = motor_data.copy() * 180 / np.pi
motor_input_size = motor_data[0].shape[0]
tot_epoch = motor_data.shape[0]
# your images in an array
sensory_data = loadImages("./data/pepper_images/")
sensory_input_size = sensory_data[0].flatten().shape[0]


# Load the sensory som
sensory_som = Som(64, sensory_input_size)
sensory_som.load('./output/som_sensory.npz')


# Load the motor som
motor_som = Som(16, motor_input_size)
motor_som.load('./output/som_motor.npz')


# Instantiate the multi modal som
som_size = 128
mmr_som = Som(som_size, 6, low=0, high=64, round_values=True)
test_percentage = 0.2
test_start_index = np.floor(tot_epoch*(1-test_percentage))
my_learning_rate = ExponentialDecay(
    starter_value=0.5,
    decay_step=test_start_index/10,
    decay_rate=0.9,
    staircase=True
)
my_radius = ExponentialDecay(
    starter_value=np.rint(som_size/3),
    decay_step=test_start_index/10,
    decay_rate=0.90,
    staircase=True
)


train_triplets = []
test_triplets = []
for epoch in range(1, tot_epoch):
    m_t_coord = motor_som.return_BMU_index(motor_data_degrees[epoch])
    s_t_coord = sensory_som.return_BMU_index(
        sensory_data[epoch-1].flatten())
    s_t1_coord = sensory_som.return_BMU_index(
        sensory_data[epoch].flatten())
    triplet = np.array([s_t_coord, m_t_coord, s_t1_coord]).flatten()

    if epoch < test_start_index:
        train_triplets.append(triplet)
        # Updating the learning rate and the radius
        learning_rate = my_learning_rate.return_decayed_value(
            global_step=epoch
        )
        radius = my_radius.return_decayed_value(global_step=epoch)
        input_vector = triplet

        # Estimating the BMU coordinates
        bmu_index = mmr_som.return_BMU_index(input_vector)
        bmu_weights = mmr_som.get_unit_weights(
            bmu_index[0], bmu_index[1])

        # Getting the BMU neighborhood
        bmu_neighborhood_list = mmr_som.return_unit_round_neighborhood(
            bmu_index[0], bmu_index[1], radius=radius
        )

        # Learning step
        mmr_som.training_single_step(
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

    else:
        test_triplets.append(triplet)

# Saving the network
file_name = output_path + "som_mmr.npz"
print("Saving the network in: " + str(file_name))
mmr_som.save(path=output_path, name="som_mmr")
np.save(output_path + "test_triplets", test_triplets)


# TODO: make mmr som, look into 3d som, look into cosine
# if __name__ == "__main__":
#    main()
