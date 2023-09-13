import numpy as np
from astropy.table import Table
from sklearn.decomposition import PCA


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


def main():
    # Specify the galaxy's GroupNumber
    galaxy_group_number = 21

    # Create an EAGLEGalaxy object for the specified galaxy
    galaxy = EAGLEGalaxy(galaxy_group_number)

    # Load data for the galaxy for all particle types
    galaxy.load_galaxy_data()

    # Now we can access the different particle types like galaxy.stars, galaxy.gas, etc.
    print(f"Stellar Particle Coordinates of Galaxy {galaxy_group_number}:")
    print(len(galaxy.stars['Coordinates'][0]))

    print(f"Gas Particle Coordinates of Galaxy {galaxy_group_number}:")
    print(len(galaxy.gas['Coordinates'][0]))

    print(f"Dark Matter Particle Coordinates of Galaxy {galaxy_group_number}:")
    print(len(galaxy.dark_matter['Coordinates'][0]))

    # Access all particles
    print(f"All Particle Coordinates of Galaxy {galaxy_group_number}:")
    all_coordinates = galaxy.get_all_coordinates()
    print(len(all_coordinates[0]))

    # Apply PCA to stars' coordinates
    stars_rotation_matrix, stars_explained_variances = EAGLEGalaxy.apply_pca_to_find_main_axes_3d(
        galaxy.stars['Coordinates'])

    # Print PCA results for stars
    print("PCA Rotation Matrix for Stars:")
    print(stars_rotation_matrix)
    print("PCA Explained Variances for Stars:")
    print(stars_explained_variances)


if __name__ == "__main__":
    main()
