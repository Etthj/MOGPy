import logging

import numpy as np
from astropy.table import Table
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO)


class EAGLEGalaxy:
    def __init__(self, galaxy_group_number, data_folder='./data'):
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
        try:
            hdu = Table.read(f'{self.data_folder}/EAGLE_Gal21.fits', format='fits')
        except FileNotFoundError:
            logging.error("Data file not found.")
            return
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
    def apply_pca_to_find_main_axes_3d(coordinates, radius=30):
        if radius is not None:
            norms = np.linalg.norm(coordinates, axis=1)
            coordinates = coordinates[norms <= radius]
            if len(coordinates) == 0:
                return None, None

        pca = PCA(n_components=3)
        pca.fit(coordinates.T)
        return pca.components_, np.sqrt(pca.explained_variance_)

    @staticmethod
    def recenter_coordinates(coordinates, center=None):
        iterations = 3 if center is None else 1
        adjusted_coords = coordinates.copy()
        medians = []

        for _ in range(iterations):
            transposed = np.array(adjusted_coords).T
            median = center if center is not None else np.median(transposed, axis=0)
            medians.append(median)
            adjusted_coords = (transposed - median).T

            dist_to_center = np.linalg.norm(adjusted_coords, axis=1)
            adaptive_radius = np.std(dist_to_center)
            core_filter = dist_to_center < adaptive_radius

            if np.count_nonzero(core_filter) >= 10:
                core_coords = adjusted_coords[core_filter]
                median = np.median(core_coords, axis=0)
                adjusted_coords -= median
            else:
                break  # Break if not enough particles are within the adaptive radius, to avoid unnecessary iterations

        return adjusted_coords, medians


def main():
    # Specify the galaxy group numbers
    galaxy_group_number = [21]

    galaxy = EAGLEGalaxy(galaxy_group_number)
    galaxy.load_galaxy_data()

    # Recenter coordinates
    stars_center_coords, _ = EAGLEGalaxy.recenter_coordinates(galaxy.stars['Coordinates'], center=None)

    # Apply PCA to stars' coordinates
    stars_rotation_matrix, stars_explained_variances = EAGLEGalaxy.apply_pca_to_find_main_axes_3d(
        stars_center_coords)

    results = galaxy_group_number, stars_rotation_matrix, stars_explained_variances, stars_center_coords

    # Create an Astropy table to store the results
    result_table = Table(names=('GroupNumber', 'ParticleType', 'RotationMatrix', 'ExplainedVariances'),
                         dtype=('i4', 'str', 'O', 'O'))

    # Populate the table with the results
    group_number, rotation_matrix, explained_variances, stars_center_coords = results
    result_table.add_row([group_number[0], 'Stars', rotation_matrix,
                          explained_variances])

    # Save the table to a file or print
    print(result_table)


if __name__ == "__main__":
    main()
