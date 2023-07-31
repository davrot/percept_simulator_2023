#%%
# DiscardElements.py
# ====================================
# removes elements from a sparse image representation
# such that a 'most uniform' coverage still exists
#
# Version V1.0, pre-07.03.2023:
#   no actual changes, is David's last code version...

import numpy as np

# assume locations is array [n, 3]
# the three entries are [shape_index, pos_x, pos_y]


def discard_elements_simple(locations: np.ndarray, target_number_elements: list):

    n_locations: int = locations.shape[0]
    locations_remain: list = []

    # Loop across all target number of elements
    for target_elem in target_number_elements:

        assert target_elem > 0, "Number of target elements must be larger than 0!"
        assert (
            target_elem <= n_locations
        ), "Number of target elements must be <= number of available locations!"

        # Build distance matrix between positions in locations_highest_res.
        # Its diagonal is defined as Inf because we don't want to consider these values in our
        # search for the minimum distances.
        distance_matrix = np.sqrt(
            ((locations[np.newaxis, :, 1:] - locations[:, np.newaxis, 1:]) ** 2).sum(
                axis=-1
            )
        )
        distance_matrix[np.arange(n_locations), np.arange(n_locations)] = np.inf

        # Find the minimal distances in upper triangle of matrix.
        idcs_remove: list = []
        while (n_locations - len(idcs_remove)) != target_elem:

            # Get index of matrix with minimal distance
            row_idcs, col_idcs = np.where(
                distance_matrix == distance_matrix[distance_matrix > 0].min()
            )

            # Get the max index.
            # It correspond to the index of the element we will remove in the locations_highest_res list
            sel_idx: int = max(row_idcs[0], col_idcs[0])
            idcs_remove.append(sel_idx)  # Save the index

            # Set current distance as Inf because we don't want to consider it further in our search
            distance_matrix[sel_idx, :] = np.inf
            distance_matrix[:, sel_idx] = np.inf

        idcs_remain: list = np.setdiff1d(np.arange(n_locations), idcs_remove)
        locations_remain.append(locations[idcs_remain, :])

    return locations_remain


# assume locations is array [n, 3]
# the three entries are [shape_index, pos_x, pos_y]
def discard_elements(
    locations: np.ndarray, target_number_elements: list, prior: np.ndarray
):

    n_locations: int = locations.shape[0]
    locations_remain: list = []
    disable_value: float = np.nan

    # if type(prior) != np.ndarray:
    #     prior = np.ones((n_locations,))
    assert prior.shape == (
        n_locations,
    ), "Prior must have same number of entries as elements in locations!"
    print(prior)

    # Loop across all target number of elements
    for target_elem in target_number_elements:

        assert target_elem > 0, "Number of target elements must be larger than 0!"
        assert (
            target_elem <= n_locations
        ), "Number of target elements must be <= number of available locations!"

        # Build distance matrix between positions in locations_highest_res.
        # Its diagonal is defined as Inf because we don't want to consider these values in our
        # search for the minimum distances.
        distance_matrix = np.sqrt(
            ((locations[np.newaxis, :, 1:] - locations[:, np.newaxis, 1:]) ** 2).sum(
                axis=-1
            )
        )
        prior_matrix = prior[np.newaxis, :] * prior[:, np.newaxis]
        distance_matrix *= prior_matrix
        distance_matrix[np.arange(n_locations), np.arange(n_locations)] = disable_value
        print(distance_matrix)

        # Find the minimal distances in upper triangle of matrix.
        idcs_remove: list = []
        while (n_locations - len(idcs_remove)) != target_elem:

            # Get index of matrix with minimal distance
            row_idcs, col_idcs = np.where(
                # distance_matrix == distance_matrix[distance_matrix > 0].min()
                distance_matrix
                == np.nanmin(distance_matrix)
            )

            # Get the max index.
            # It correspond to the index of the element we will remove in the locations_highest_res list
            print(row_idcs[0], col_idcs[0])
            # if prior[row_idcs[0]] >= prior[col_idcs[0]]:
            #     sel_idx = row_idcs[0]
            # else:
            #     sel_idx = col_idcs[0]
            d_row = np.nansum(distance_matrix[row_idcs[0], :])
            d_col = np.nansum(distance_matrix[:, col_idcs[0]])
            if d_row > d_col:
                sel_idx = col_idcs[0]
            else:
                sel_idx = row_idcs[0]
            # sel_idx: int = max(row_idcs[0], col_idcs[0])
            idcs_remove.append(sel_idx)  # Save the index

            # Set current distance as Inf because we don't want to consider it further in our search
            distance_matrix[sel_idx, :] = disable_value
            distance_matrix[:, sel_idx] = disable_value

        idcs_remain: list = np.setdiff1d(np.arange(n_locations), idcs_remove)
        locations_remain.append(locations[idcs_remain, :])

    return locations_remain


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # generate a circle with n locations
    n_locations: int = 20
    phi = np.arange(n_locations) / n_locations * 2 * np.pi
    locations = np.ones((n_locations, 3))
    locations[:, 1] = np.cos(phi)
    locations[:, 2] = np.sin(phi)
    prior = np.ones((n_locations,))
    prior[:10] = 0.1
    locations_remain = discard_elements(locations, [n_locations // 5], prior=prior)

    plt.plot(locations[:, 1], locations[:, 2], "ko")
    plt.plot(locations_remain[0][:, 1], locations_remain[0][:, 2], "rx")
    plt.show()

    locations_remain_simple = discard_elements_simple(locations, [n_locations // 5])

    plt.plot(locations[:, 1], locations[:, 2], "ko")
    plt.plot(locations_remain_simple[0][:, 1], locations_remain_simple[0][:, 2], "rx")
    plt.show()


# %%
