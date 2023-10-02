from MOGPy.MOGPy_app import EAGLEGalaxy
import unittest
from unittest.mock import patch
import numpy as np


class TestEAGLEGalaxy(unittest.TestCase):

    def setUp(self):
        self.galaxy = EAGLEGalaxy(21)

        np.random.seed(0)
        n_samples = 1000

        # First distribution
        mean1 = np.array([0, 0, 0])
        covariance_matrix1 = np.array([
            [1.0, 0.8, 0.5],
            [0.8, 1.0, 0.3],
            [0.5, 0.3, 1.0]
        ])
        coordinates1 = np.random.multivariate_normal(mean1, covariance_matrix1, n_samples).T

        # Second distribution
        mean2 = np.array([1, 1, 1])
        covariance_matrix2 = np.array([
            [1.1, 0.9, 0.6],
            [0.9, 1.2, 0.4],
            [0.6, 0.4, 0.9]
        ])
        coordinates2 = np.random.multivariate_normal(mean2, covariance_matrix2, 100).T

        # Third distribution
        mean3 = np.array([-0.5, -0.5, -0.5])
        covariance_matrix3 = np.array([
            [0.9, 0.7, 0.4],
            [0.7, 0.8, 0.5],
            [0.4, 0.5, 1.1]
        ])
        coordinates3 = np.random.multivariate_normal(mean3, covariance_matrix3, 100).T

        # Concatenating all distributions to form one coordinates variable
        self.coordinates = np.concatenate((coordinates1, coordinates2, coordinates3), axis=1)

    @patch('MOGPy.MOGPy_app.Table.read')
    def test_load_galaxy_data_file_not_found(self, mock_read):
        mock_read.side_effect = FileNotFoundError
        with self.assertLogs(level='ERROR') as cm:
            self.galaxy.load_galaxy_data()
        self.assertEqual(cm.output, ['ERROR:root:Data file not found.'])

    def test_recenter_coordinates(self):
        self.recentered_coordinates, self.medians = EAGLEGalaxy.recenter_coordinates(self.coordinates)

        # Validate that the median of recentered_coordinates is close to [0, 0, 0]
        median_of_recentered = np.median(self.recentered_coordinates, axis=1)
        np.testing.assert_allclose(median_of_recentered, np.zeros(3), atol=1e-6,
                                   err_msg="Median of recentered coordinates is not close to [0, 0, 0]")

        expected_medians = []
        adjusted_coords = self.coordinates.copy()
        for _ in range(3):
            median = np.median(adjusted_coords, axis=1)
            expected_medians.append(median)
            adjusted_coords -= median.reshape(-1, 1)

        self.assertTrue(np.all(np.isclose(expected_medians[0], [0, 0, 0], atol=1e-1)))

        # Assert that pairwise distances are preserved
        pairwise_distances_original = np.linalg.norm(self.coordinates[:, :, None] - self.coordinates[:, None, :],
                                                     axis=0)
        pairwise_distances_recentered = np.linalg.norm(
            self.recentered_coordinates[:, :, None] - self.recentered_coordinates[:, None, :], axis=0)
        np.testing.assert_allclose(pairwise_distances_original, pairwise_distances_recentered, atol=1e-6,
                                   err_msg="Pairwise distances are not preserved")

        # Assert the variance is preserved along each dimension
        original_variances = np.var(self.coordinates, axis=1)
        recentered_variances = np.var(self.recentered_coordinates, axis=1)
        np.testing.assert_allclose(original_variances, recentered_variances, atol=1e-6,
                                   err_msg="Variances are not preserved along each dimension")

    def test_apply_pca_to_find_main_axes_3d(self):
        self.recentered_coordinates, _ = EAGLEGalaxy.recenter_coordinates(self.coordinates)

        components, explained_var = EAGLEGalaxy.apply_pca_to_find_main_axes_3d(self.recentered_coordinates, None)

        # Asserting the correctness of components and explained_var
        covariance_matrix = np.cov(self.recentered_coordinates)
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        # Sorting eigenvalues and eigenvectors by eigenvalue in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]

        # Asserting that the PCA components are correctly found
        np.testing.assert_allclose(np.abs(components), np.abs(sorted_eigenvectors.T), atol=1e-6,
                                   err_msg="PCA components are incorrect")

        # Asserting that the explained variances are correctly found
        np.testing.assert_allclose(explained_var, np.sqrt(sorted_eigenvalues), atol=1e-6,
                                   err_msg="Explained variances are incorrect")

        # Asserting that the explained variances are in decreasing order
        self.assertTrue(np.all(np.diff(explained_var) <= 0), "Explained variances are not in decreasing order")


if __name__ == '__main__':
    unittest.main()
