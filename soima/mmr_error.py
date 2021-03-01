import numpy as np
from pyERA.som import Som


# Load MMR some and test data
mmr_som = Som(64, 6)
mmr_som.load('./output/som_mmr.npz')
test_triplets = np.load('./output/test_triplets.npy')


errors = []
for triplet in test_triplets:

    input_vector = triplet.copy()

    # Estimating the BMU coordinates
    bmu_index = mmr_som.return_BMU_index(input_vector)
    bmu_weights = mmr_som.get_unit_weights(
        bmu_index[0], bmu_index[1]
    )
    errors.append(np.linalg.norm(input_vector - bmu_weights))


print(f"\nAverage Error on Test Set - Norm: {np.mean(errors)}\n")
