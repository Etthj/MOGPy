# New Branch of the Code

This GitHub repository contains the new branch of MOGPy code, which is currently under development.

## Objective

The goal of this new branch is to make significant improvements to the existing MOGPy code, focusing on the following points:

- Code refactoring using an object-oriented approach.
- Integration of unit tests to ensure code quality and reliability.
- Comply with PEP8 standards and eliminate major errors.
- Exploring the use of multiprocessing to make code more scalable.

## Current Progress

At this point, the new branch of MOGPy code is under development, and the main code is contained in the `MOGPy_app.py` file.
It is important to note that the current code is a rework that may not include all the functionality of the original code due to time constraints. 
However, the main goal is to provide quality and functional code.

Refine of the `recenter_coordinates` funtion:

1. **Adjusting Radius of Influence:**
During each iteration, gradually reduce the radius within which we consider particles for recalculating the median/center. This will help in reducing the influence of edge structures.

2. **Adaptive Core Selection:**
Instead of a hard threshold like np.max(adjusted_coords) / 10, I made it adaptive based on some statistical properties of the distribution, such as standard deviation, so that it adapts to the scale of the structures in the distribution.

3. **Convergence Criteria:**
I implemented a convergence criterion to stop the iterations when the center does not change significantly between iterations.

Refine of the `apply_pca_to_find_main_axes_3d` function:

1. **Selection Radius:**
I introduced a radius parameter to filter the coordinates, allowing the PCA to focus only on particles within a certain distance from the median center. This helps in removing outliers or points far away from the dense regions before applying PCA, providing a more accurate representation of the main axes of the distribution’s core.

2. **Empty Radius Handling:**
I included a mechanism to handle scenarios where no points exist within the specified radius, returning suitable values (`None`) to indicate the absence of valid points for PCA.

Include Unit tests:

1. Validate that the median of recentered_coordinates is close to [0, 0, 0]
2. Assert that pairwise distances are preserved during recentered operations
3. Assert the variance is preserved along each dimension during recentered operations
4. Asserting the correctness of components and explained_var in the PCA operation
5. Asserting that the explained variances are in decreasing order