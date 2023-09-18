import unittest

import numpy as np
from astropy.table import Table
from sklearn.decomposition import PCA
from multiprocessing import Pool


class EAGLEGalaxy:
    def __init__(self, galaxy_group_number, data_folder='../data'):
        self.galaxy_group_number = galaxy_group_number
        self.data_folder = data_folder
        self.particles = {
            0: 'gas',
            1: 'dark_matter',
            4: 'stars'
        }
        self.stars = None
        self.gas = None
        self.dark_matter = None
        self.all_particles = None

    def load_galaxy_data(self):
        # Read the snapshot data into a table
        hdu = Table.read(f'data/EAGLE_RefL0012N0188_Snapshot23.fits', format='fits')
        all_particle_data = {}  # Create a dictionary to store data for all particle types
        for particle_type, particle_name in self.particles.items():
            mask = (hdu['GroupNumber'] == self.galaxy_group_number) & (hdu['itype'] == particle_type)
            particle_data = {
                'Coordinates': np.array([hdu['Coordinates_x'][mask],
                                         hdu['Coordinates_y'][mask],
                                         hdu['Coordinates_z'][mask]]),
                'Mass': hdu['Mass'][mask]
            }
            all_particle_data[particle_name] = particle_data  # Store the data for this particle type

            # Assign the loaded data to the corresponding attribute
            setattr(self, particle_name.lower(), particle_data)

        # Assign all_particle_data to the all_particles attribute
        self.all_particles = all_particle_data

    def get_all_coordinates(self):
        all_coordinates = np.concatenate([data['Coordinates'] for data in self.all_particles.values()], axis=1)
        return all_coordinates

    @staticmethod
    def apply_pca_to_find_main_axes_3d(coordinates):
        pca = PCA(n_components=3)
        pca.fit(coordinates.T)
        return pca.components_, np.sqrt(pca.explained_variance_)


def process_galaxy(galaxy_group_number):
    galaxy = EAGLEGalaxy(galaxy_group_number)
    galaxy.load_galaxy_data()

    # Apply PCA to stars' coordinates
    stars_rotation_matrix, stars_explained_variances = EAGLEGalaxy.apply_pca_to_find_main_axes_3d(
        galaxy.stars['Coordinates'])

    return galaxy_group_number, stars_rotation_matrix, stars_explained_variances


class TestEAGLEGalaxyProcessing(unittest.TestCase):
    def test_process_galaxy(self):
        # Test the process_galaxy function with a known input
        galaxy_group_number = 21
        result = process_galaxy(galaxy_group_number)

        # Extract the expected result
        _, expected_rotation_matrix, expected_explained_variances = process_galaxy(galaxy_group_number)

        # Assert that the result is as expected
        self.assertEqual(result[0], galaxy_group_number)
        self.assertTrue(np.array_equal(result[1], expected_rotation_matrix))
        self.assertTrue(np.array_equal(result[2], expected_explained_variances))


def main():
    # Specify the galaxy group numbers you want to process
    galaxy_group_numbers = [21, 21, 21]  # Add more group numbers as needed

    # Create a multiprocessing pool with the number of processes you want to use
    pool = Pool(processes=len(galaxy_group_numbers))

    # Use the pool to process galaxy group numbers in parallel
    results = pool.map(process_galaxy, galaxy_group_numbers)
    pool.close()
    pool.join()

    # Create an Astropy table to store the results
    result_table = Table(names=('GroupNumber', 'RotationMatrix', 'ExplainedVariances'),
                         dtype=('i4', 'O', 'O'))

    # Populate the table with the results
    for group_number, rotation_matrix, explained_variances in results:
        result_table.add_row([group_number, rotation_matrix, explained_variances])

    # Save the table to a file or perform any other desired operations
    print(result_table)


if __name__ == "__main__":
    main()
