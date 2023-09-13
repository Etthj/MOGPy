# This file is part of MOGPy: Python package for the study of galaxy morphology in universe simulations.
#
# MOGPy is free software: redistribution and use in source and binary forms,
# with or without modification, are permitted.

import math

import numpy as np
from sklearn.decomposition import PCA


def apply_pca_to_find_main_axes_3d(coordinates):
    """
    Applies Principal Component Analysis (PCA) to find the main axes in a three-dimensional space.
    
    Parameters:
    - coordinates: A 2D numpy array or a list of lists containing the coordinates of points in three-dimensional space.
                   Each row represents a point, and each column represents a coordinate (x, y, z).
                   
    Returns: - rotation_matrix: A 2D numpy array representing the rotation matrix obtained from PCA. Each row
    corresponds to a principal component (main axis), and each column corresponds to a coordinate axis. -
    explained_variances: A 1D numpy array containing the explained variances along the principal components. The
    values indicate the amount of variance captured by each principal component.
    """
    m = np.array(coordinates).T
    pca = PCA(3)
    pca.fit(m)
    rotation_matrix = pca.components_
    a, b, c = np.sqrt(pca.explained_variance_[0]), np.sqrt(pca.explained_variance_[1]), np.sqrt(
        pca.explained_variance_[2])
    return rotation_matrix, a, b, c


def pca_2d(coordinates):
    """
    Retrieves the axis ratios of 3 projections for a given 3D distribution and the projected angle of the main axes.
    
    Parameters:
    - coordinates: A 2D numpy array or a list of lists containing the coordinates of points in three-dimensional space.
                   Each row represents a point, and each column represents a coordinate (x, y, z).
                   
    Returns:
    - xy_a, xy_b: The square roots of the explained variances along the x and y axes in the xy projection.
    - yz_b, yz_c: The square roots of the explained variances along the y and z axes in the yz projection.
    - xz_a, xz_c: The square roots of the explained variances along the x and z axes in the xz projection.
    - angle_1: The projected angle of the main axes in the xy projection (in degrees).
    - angle_2: The projected angle of the main axes in the yz projection (in degrees).
    - angle_3: The projected angle of the main axes in the xz projection (in degrees).
    """
    m = np.array([coordinates[:, 0], coordinates[:, 1]]).T
    pca = PCA(n_components=2)
    pca.fit_transform(m)
    xy_a, xy_b = np.sqrt(pca.explained_variance_[0]), np.sqrt(pca.explained_variance_[1])
    angle_1 = np.degrees(math.atan2(pca.components_[0, 1], pca.components_[0, 0]))
    angle_1 = angle_1 + 180 if angle_1 < -90 else angle_1
    angle_1 = angle_1 - 180 if angle_1 > 90 else angle_1

    n = np.array([coordinates[:, 1], coordinates[:, 2]]).T
    pca.fit_transform(n)
    yz_b, yz_c = np.sqrt(pca.explained_variance_[0]), np.sqrt(pca.explained_variance_[1])
    angle_2 = np.degrees(math.atan2(pca.components_[0, 1], pca.components_[0, 0]))
    angle_2 = angle_2 + 180 if angle_2 < -90 else angle_2
    angle_2 = angle_2 - 180 if angle_2 > 90 else angle_2

    o = np.array([coordinates[:, 0], coordinates[:, 2]]).T
    pca.fit_transform(o)
    xz_a, xz_c = np.sqrt(pca.explained_variance_[0]), np.sqrt(pca.explained_variance_[1])
    angle_3 = np.degrees(math.atan2(pca.components_[0, 1], pca.components_[0, 0]))
    angle_3 = angle_3 + 180 if angle_3 < -90 else angle_3
    angle_3 = angle_3 - 180 if angle_3 > 90 else angle_3

    return xy_a, xy_b, yz_b, yz_c, xz_a, xz_c, angle_1, angle_2, angle_3


def select_core_and_apply_method(coordinates, density_radius, _2d):
    """
    Selects particles within a density radius from the center and applies a method based on the given parameters.
    
    Parameters:
    - coordinates: A 3D numpy array containing the coordinates of particles.
                   Each row represents a point, and each column represents a coordinate (x, y, z).
    - density_radius: The radius within which particles are considered part of the core.
    - _2D: A boolean value indicating whether to perform a 2D analysis or 3D analysis.
    
    Returns:
    If _2D is True:
    - core_xy_a: The square root of the explained variance along the x-axis in the XY projection of the core particles.
    - core_xy_b: The square root of the explained variance along the y-axis in the XY projection of the core particles.
    - core_yz_b: The square root of the explained variance along the y-axis in the YZ projection of the core particles.
    - core_yz_c: The square root of the explained variance along the z-axis in the YZ projection of the core particles.
    - core_xz_a: The square root of the explained variance along the x-axis in the XZ projection of the core particles.
    - core_xz_c: The square root of the explained variance along the z-axis in the XZ projection of the core particles.
    - angle_1: The projected angle of the main axes in the XY projection of the core particles (in degrees).
    - angle_2: The projected angle of the main axes in the YZ projection of the core particles (in degrees).
    - angle_3: The projected angle of the main axes in the XZ projection of the core particles (in degrees).
    
    If _2D is False:
    - core_rotation_matrix: The rotation matrix obtained from PCA for the core particles.
                            It represents the main axes in the three-dimensional space.
    - core_a: The square root of the explained variance along the first principal component of the core particles.
    - core_b: The square root of the explained variance along the second principal component of the core particles.
    - core_c: The square root of the explained variance along the third principal component of the core particles.
    """
    distance_to_center = np.sqrt(coordinates[0] ** 2 + coordinates[1] ** 2 + coordinates[2] ** 2)
    core_selected_particles = np.array(
        [coordinates[0][distance_to_center < density_radius], coordinates[1][distance_to_center < density_radius],
         coordinates[2][distance_to_center < density_radius]])
    if _2d:
        core_xy_a, core_xy_b, core_yz_b, core_yz_c, core_xz_a, core_xz_c, angle_1, angle_2, angle_3 = pca_2d(
            core_selected_particles)
        return core_xy_a, core_xy_b, core_yz_b, core_yz_c, core_xz_a, core_xz_c, angle_1, angle_2, angle_3
    else:
        core_rotation_matrix, core_a, core_b, core_c = apply_pca_to_find_main_axes_3d(core_selected_particles)
        return core_rotation_matrix, core_a, core_b, core_c


def set_origin(coordinates, centre_of_mass):
    """
    Sets the origin of the coordinate system based on the median of a 3D distribution of points.
    
    Parameters:
    - coordinates: A 3D numpy array containing the coordinates of points.
                   Each row represents a point, and each column represents a coordinate (x, y, z).
    - CentreOfMass: The coordinates of the center of mass. If provided, it will be used as the initial median estimate.
    
    Returns:
    - coordinates_new: A 3D numpy array containing the updated coordinates with respect to the new origin.
    - median_list: A list containing the medians for each pass.
    """
    passes = 3 if centre_of_mass is None else 1
    coordinates_new = coordinates.copy()
    median_list = []

    for i in range(passes):
        m = np.array(coordinates_new).T
        if centre_of_mass is not None:
            median = centre_of_mass
        else:
            median = np.median(m, axis=0)

        median_list.append(median)
        coordinates_t = m - median
        coordinates_new = coordinates_t.T

        distance_to_center = np.sqrt(coordinates_new[0] ** 2 + coordinates_new[1] ** 2 + coordinates_new[2] ** 2)
        core = np.max(coordinates_new) / 10
        m = coordinates_new.T
        n = np.array([coordinates_new[0][distance_to_center < core],
                      coordinates_new[1][distance_to_center < core],
                      coordinates_new[2][distance_to_center < core]]).T
        if len(n.T[0]) < 10:
            n = coordinates_new.T
        median = np.median(n, axis=0)
        coordinates_1_t = m - median
        coordinates_new = coordinates_1_t.T

    return coordinates_new, median_list


def find_density_radius(coordinates, density_core, fraction_density, masses):
    """
    Estimates the density radius of a 3D distribution of points by computing a mass growth curve over the distribution.
    This function tries to take into account the diversity of galaxy structures in universe simulations, 
    which regularly leads to difficulties in the density calculation. 
    
    Parameters:
    - coordinates: A 3D numpy array containing the coordinates of points.
                   Each row represents a point, and each column represents a coordinate (x, y, z).
    - density_core: The desired density in the core region, ranging from 0 to 1.
    - fraction_density: The desired fraction of the maximum density, ranging from 0 to 1.
    - masses: A numpy array containing the masses of the points corresponding to the coordinates.
    - gn: The GroupNumber of the distribution.
    
    Returns:
    - density_radius: The estimated density radius.
    """
    maxi = 50
    radius = []
    mass = []
    distance_to_center = np.sqrt(coordinates[0] ** 2 + coordinates[1] ** 2 + coordinates[2] ** 2)
    maximum = np.max(np.array([np.max(coordinates[0]), np.max(coordinates[1]), np.max(coordinates[2])]))
    k = 0

    for i in np.linspace(0, 2 * maximum, maxi):
        if i == 0:
            mass.append(0)
            radius.append(0)
        else:
            j = i
            radius.append(j)
            mass.append(np.sum(masses[distance_to_center <= j]))
        if k > 3:
            mean_3_last_points = np.sum((mass[k - 3] + mass[k - 2] + mass[k - 1]) / 3)
            if abs(mass[k] - mean_3_last_points) <= 1 * np.std([mass[k - 3], mass[k - 2], mass[k - 1]]):
                break
        k = k + 1
    mass_core = (1 - density_core) * mass[-1]

    if mass_core < 1e6:
        core_radius = 0.003
    else:
        core_radius, _ = interpolated_intercept(np.array(radius), np.array(mass) / mass[-1],
                                                np.linspace(mass_core / mass[-1], mass_core / mass[-1], len(radius)))

    radius = []
    density = []
    k = 0
    for i in np.linspace(core_radius, 2 * maximum + core_radius, maxi):
        j = i
        radius.append(j)
        mass = np.sum(masses[distance_to_center < j])
        volume = 4 / 3 * np.pi * j ** 3
        density.append(mass / volume)
        if k > 3:
            if density[0] == 0:
                density[0] = np.NaN
            if np.std([density[k - 3] / density[0], density[k - 2] / density[0], density[k - 1] / density[0]]) < 1e-4:
                break
        k = k + 1

    if np.isnan(density[0]):
        density_radius = 2 * 0.03
    else:
        if np.max(density / density[0]) > 1:
            density_radius = 2 * 0.03
        else:
            if density[-1] / density[0] > 0.2:
                density_radius = radius[-1]
            else:
                x, y_ = interpolated_intercept(np.array(radius), np.array(density / density[0]),
                                               np.linspace(fraction_density, fraction_density, len(radius)))
                density_radius = x
        if density_radius < 0.03:
            density_radius = 2 * 0.03

    return density_radius


def ite_for_axis_ratios_3d(coordinates, core_a, core_b, core_c, ite, density_radius):
    """
    Iteratively estimates the axis ratios of a 3D distribution of points.
    
    Parameters:
    - coordinates: A 3D numpy array containing the coordinates of points.
                   Each row represents a point, and each column represents a coordinate (x, y, z).
    - core_a: Initial value for the length of the major axis.
    - core_b: Initial value for the length of the intermediate axis.
    - core_c: Initial value for the length of the minor axis.
    - ite: The number of iterations to perform.
    - gn: The GroupNumber of the distribution.
    - density_radius: The density radius.
    
    Returns:
    - axis_a: A list containing the lengths of the major axis at each iteration.
    - axis_b: A list containing the lengths of the intermediate axis at each iteration.
    - axis_c: A list containing the lengths of the minor axis at each iteration.
    """
    axis_a, axis_b, axis_c = [], [], []

    for i in range(ite):
        core_a, core_b, core_c = passe_2_3d(coordinates, core_a, core_b, core_c, density_radius)
        axis_a.append(density_radius)
        axis_b.append(density_radius * core_b / core_a)
        axis_c.append(density_radius * core_c / core_a)

        if i > 3:
            if np.std(axis_c[i - 3:i]) == 0:
                break

    return axis_a[-1], axis_b[-1], axis_c[-1]


def ite_for_axis_ratios_2d(coordinates, core_xy_a, core_xy_b, core_yz_b, core_yz_c, core_xz_a, core_xz_c, angle_1,
                           angle_2, angle_3, ite, density_radius):
    """
    Iteratively estimates the axis ratios of a 2D distribution of points.
    
    Parameters:
    - coordinates: A 2D numpy array containing the coordinates of points.
                   Each row represents a point, and each column represents a coordinate (x, y).
    - core_xy_a: Initial value for the length of the major axis in the xy-plane.
    - core_xy_b: Initial value for the length of the minor axis in the xy-plane.
    - core_yz_b: Initial value for the length of the major axis in the yz-plane.
    - core_yz_c: Initial value for the length of the minor axis in the yz-plane.
    - core_xz_a: Initial value for the length of the major axis in the xz-plane.
    - core_xz_c: Initial value for the length of the minor axis in the xz-plane.
    - angle_1: Initial angle of rotation for the xy-plane.
    - angle_2: Initial angle of rotation for the yz-plane.
    - angle_3: Initial angle of rotation for the xz-plane.
    - ite: The number of iterations to perform.
    - gn: The GroupNumber of the distribution.
    - density_radius: The density radius.
    
    Returns:
    - axis_xy_a: A list containing the lengths of the major axis in the xy-plane at each iteration.
    - axis_xy_b: A list containing the lengths of the minor axis in the xy-plane at each iteration.
    - axis_yz_b: A list containing the lengths of the major axis in the yz-plane at each iteration.
    - axis_yz_c: A list containing the lengths of the minor axis in the yz-plane at each iteration.
    - axis_xz_a: A list containing the lengths of the major axis in the xz-plane at each iteration.
    - axis_xz_c: A list containing the lengths of the minor axis in the xz-plane at each iteration.
    """
    axis_xy_a, axis_xy_b, axis_yz_b, axis_yz_c, axis_xz_a, axis_xz_c = [], [], [], [], [], []

    for i in range(ite):
        core_xy_a, core_xy_b, core_yz_b, core_yz_c, core_xz_a, core_xz_c = passe_2_2d(coordinates, core_xy_a, core_xy_b,
                                                                                      core_yz_b, core_yz_c, core_xz_a,
                                                                                      core_xz_c, angle_1, angle_2,
                                                                                      angle_3, density_radius)
        axis_xy_a.append(density_radius)
        axis_xy_b.append(density_radius * core_xy_b / core_xy_a)
        axis_yz_b.append(density_radius)
        axis_yz_c.append(density_radius * core_yz_c / core_yz_b)
        axis_xz_a.append(density_radius)
        axis_xz_c.append(density_radius * core_xz_c / core_xz_a)

        if i > 2:
            if np.std(axis_xy_a[i - 3:i]) == 0:
                break

    return axis_xy_a, axis_xy_b, axis_yz_b, axis_yz_c, axis_xz_a


def pca_2d_passe2(coordinates):
    """
    Retrieves the axis ratios of the XY projection for a given 3D distribution using PCA.
    
    Parameters:
    - coordinates: A 2D numpy array or a list of lists containing the coordinates of points in three-dimensional space.
                   Each row represents a point, and each column represents a coordinate (x, y, z).
                   
    Returns:
    - xy_a: The square root of the explained variance along the x-axis in the XY projection.
    - xy_b: The square root of the explained variance along the y-axis in the XY projection.
    """
    pca = PCA(n_components=2)
    pca.fit_transform(coordinates[:, :2])
    xy_a, xy_b = np.sqrt(pca.explained_variance_[0]), np.sqrt(pca.explained_variance_[1])

    return xy_a, xy_b


def passe_2_3d(coordinates, core_a, core_b, core_c, density_radius):
    """
    Updates the axis lengths of a 3D distribution of points based on a density radius.
    
    Parameters:
    - coordinates: A 3D numpy array containing the coordinates of points.
                   Each row represents a point, and each column represents a coordinate (x, y, z).
    - core_a: Length of the major axis.
    - core_b: Length of the intermediate axis.
    - core_c: Length of the minor axis.
    - gn: The GroupNumber of the distribution.
    - density_radius: The density radius.
    
    Returns:
    - a: Updated length of the major axis.
    - b: Updated length of the intermediate axis.
    - c: Updated length of the minor axis.
    """
    ratio_ab = core_a / core_b
    ratio_ac = core_a / core_c

    a = density_radius
    b = a / ratio_ab
    c = a / ratio_ac

    distance_to_center = ellipse_equation_3d(coordinates[0], coordinates[1], coordinates[2], a, b, c)
    core_selected_particles = np.array([coordinates[0][distance_to_center < 1], coordinates[1][distance_to_center < 1],
                                        coordinates[2][distance_to_center < 1]])

    if len(core_selected_particles[0]) < 10:
        core_selected_particles = coordinates

    _, a, b, c = apply_pca_to_find_main_axes_3d(core_selected_particles)

    return a, b, c


def passe_2_2d(coordinates, core_xy_a, core_xy_b, core_yz_b, core_yz_c, core_xz_a, core_xz_c, angle_1, angle_2, angle_3,
               density_radius):
    """
    Updates the axis lengths of a 2D distribution of points based on a density radius.
    
    Parameters:
    - coordinates: A 2D numpy array containing the coordinates of points.
                   Each row represents a point, and each column represents a coordinate (x, y).
    - core_xy_a: Length of the major axis in the xy plane.
    - core_xy_b: Length of the minor axis in the xy plane.
    - core_yz_b: Length of the major axis in the yz plane.
    - core_yz_c: Length of the minor axis in the yz plane.
    - core_xz_a: Length of the major axis in the xz plane.
    - core_xz_c: Length of the minor axis in the xz plane.
    - angle_1: Angle of rotation for the xy plane.
    - angle_2: Angle of rotation for the yz plane.
    - angle_3: Angle of rotation for the xz plane.
    - gn: The GroupNumber of the distribution.
    - density_radius: The density radius.
    
    Returns:
    - Updated values for core_xy_a, core_xy_b, core_yz_b, core_yz_c, core_xz_a, core_xz_c.
    """
    ratio_xy_ab = core_xy_a / core_xy_b
    ratio_yz_bc = core_yz_b / core_yz_c
    ratio_xz_ac = core_xz_a / core_xz_c

    xy_a = density_radius
    xy_b = xy_a / ratio_xy_ab
    distance_to_center_xy = ellipse_equation(coordinates[0], coordinates[1], angle_1, xy_a, xy_b)
    core_selected_particles_xy = np.array(
        [coordinates[0][distance_to_center_xy < 1], coordinates[1][distance_to_center_xy < 1]])

    if len(core_selected_particles_xy[0]) < 10:
        core_selected_particles_xy = coordinates

    core_xy_a, core_xy_b = pca_2d_passe2(core_selected_particles_xy)

    yz_b = density_radius
    yz_c = yz_b / ratio_yz_bc
    distance_to_center_yz = ellipse_equation(coordinates[1], coordinates[2], angle_2, yz_b, yz_c)
    core_selected_particles_yz = np.array(
        [coordinates[1][distance_to_center_yz < 1], coordinates[2][distance_to_center_yz < 1]])

    if len(core_selected_particles_yz[0]) < 10:
        core_selected_particles_yz = coordinates

    core_yz_b, core_yz_c = pca_2d_passe2(core_selected_particles_yz)

    xz_a = density_radius
    xz_c = xz_a / ratio_xz_ac
    distance_to_center_xz = ellipse_equation(coordinates[0], coordinates[2], angle_3, xz_a, xz_c)
    core_selected_particles_xz = np.array(
        [coordinates[0][distance_to_center_xz < 1], coordinates[2][distance_to_center_xz < 1]])

    if len(core_selected_particles_xz[0]) < 10:
        core_selected_particles_xz = coordinates

    core_xz_a, core_xz_c = pca_2d_passe2(core_selected_particles_xz)

    return core_xy_a, core_xy_b, core_yz_b, core_yz_c, core_xz_a, core_xz_c


def to_align_coordinates(coordinates, core_rotation_matrix):
    m = np.array(coordinates).T
    align_general_coordinates_tab = (core_rotation_matrix @ m.T).T
    align_coordinates = align_general_coordinates_tab.T
    return align_coordinates


def ellipse_equation_3d(x, y, z, a, b, c):
    return np.sqrt(x ** 2 / a ** 2 + y ** 2 / b ** 2 + z ** 2 / c ** 2)


def ellipse_equation(x, y, angle, a, b):
    angle = np.radians(angle)
    return np.sqrt(
        (x * np.cos(angle) + y * np.sin(angle)) ** 2 / a ** 2 + (x * np.sin(angle) - y * np.cos(angle)) ** 2 / b ** 2)


def growth_curve3d(coordinates, masses, a, b, c):
    """
    Estimates the growth curve of a 3D distribution of points by computing the mass growth over the distribution.

    Parameters:
        coordinates (list): List of coordinate arrays representing the x, y, and z coordinates of the points.
        masses (list): List of mass arrays corresponding to the points.
        a (float): Semi-major axis length of the ellipsoid.
        b (float): Semi-intermediate axis length of the ellipsoid.
        c (float): Semi-minor axis length of the ellipsoid.

    Returns:
        tuple: A tuple containing the following elements:
            - mass (list): List of mass values corresponding to each radius.
            - radius (list): List of radius values.
            - h50_radius (float): Radius at which the mass fraction is 0.5.
            - h20_radius (float): Radius at which the mass fraction is 0.2.
            - h80_radius (float): Radius at which the mass fraction is 0.8.
            - concentration (float): Concentration parameter calculated as 5 * log10(h80_radius / h20_radius).
    """
    ba = b / a
    ca = c / a
    maxi = 50
    radius = []
    mass = []
    k = 0
    maximum = np.max(np.array([np.max(coordinates[0]), np.max(coordinates[1]), np.max(coordinates[2])]))

    for i in np.linspace(0, 2 * maximum, maxi):
        if i == 0:
            mass.append(0)
            radius.append(0)
        else:
            j = i
            radius.append(j)
            distance = np.array(ellipse_equation_3d(coordinates[0], coordinates[1], coordinates[2], j, ba * j, ca * j))
            mass.append(np.sum(masses[distance < 1]))
        if k > 3:
            mean_3_last_points = np.sum((mass[k - 3] + mass[k - 2] + mass[k - 1]) / 3)
            if abs(mass[k] - mean_3_last_points) <= 1 * np.std([mass[k - 3], mass[k - 2], mass[k - 1]]):
                break
        k = k + 1
    if mass[-1] < 1e2:
        h50_radius, h20_radius, h80_radius, concentration = np.NaN, np.NaN, np.NaN, np.NaN
    else:
        h50_radius, _ = interpolated_intercept(np.array(radius), np.array(mass) / mass[-1],
                                               np.linspace(0.5, 0.5, len(mass)))
        h80_radius, _ = interpolated_intercept(np.array(radius), np.array(mass) / mass[-1],
                                               np.linspace(0.8, 0.8, len(mass)))
        h20_radius, _ = interpolated_intercept(np.array(radius), np.array(mass) / mass[-1],
                                               np.linspace(0.2, 0.2, len(mass)))
        concentration = 5 * np.log10(h80_radius / h20_radius)
    return mass, radius, h50_radius, h20_radius, h80_radius, concentration


def growth_curve_2d(coordinates, masses, xy_a, xy_b, yz_b, yz_c, xz_a, xz_c, angle_1, angle_2, angle_3):
    """
    Computes the growth curve for 2D distributions in three orthogonal planes: XY, YZ, and XZ.

    Parameters:
        coordinates (list): List of coordinate arrays representing the x, y, and z coordinates of the points.
        masses (list): List of mass arrays corresponding to the points.
        xy_a (float): Semi-major axis length of the ellipse in the XY plane.
        xy_b (float): Semi-minor axis length of the ellipse in the XY plane.
        yz_b (float): Semi-minor axis length of the ellipse in the YZ plane.
        yz_c (float): Semi-major axis length of the ellipse in the YZ plane.
        xz_a (float): Semi-major axis length of the ellipse in the XZ plane.
        xz_c (float): Semi-minor axis length of the ellipse in the XZ plane.
        angle_1 (float): Angle of rotation for the ellipse in the XY plane.
        angle_2 (float): Angle of rotation for the ellipse in the YZ plane.
        angle_3 (float): Angle of rotation for the ellipse in the XZ plane.

    Returns: tuple: A tuple containing the following elements: - h50_radius_xy (float): Radius at which the mass
    fraction is 0.5 in the XY plane. - h20_radius_xy (float): Radius at which the mass fraction is 0.2 in the XY
    plane. - h80_radius_xy (float): Radius at which the mass fraction is 0.8 in the XY plane. - concentration_xy (
    float): Concentration parameter calculated as 5 * log10(h80_radius_xy / h20_radius_xy) in the XY plane. -
    h50_radius_yz (float): Radius at which the mass fraction is 0.5 in the YZ plane. - h20_radius_yz (float): Radius
    at which the mass fraction is 0.2 in the YZ plane. - h80_radius_yz (float): Radius at which the mass fraction is
    0.8 in the YZ plane. - concentration_yz (float): Concentration parameter calculated as 5 * log10(h80_radius_yz /
    h20_radius_yz) in the YZ plane. - h50_radius_xz (float): Radius at which the mass fraction is 0.5 in the XZ
    plane. - h20_radius_xz (float): Radius at which the mass fraction is 0.2 in the XZ plane. - h80_radius_xz (
    float): Radius at which the mass fraction is 0.8 in the XZ plane. - concentration_xz (float): Concentration
    parameter calculated as 5 * log10(h80_radius_xz / h20_radius_xz) in the XZ plane.
    """
    maxi = 50
    maximum = np.max(np.array([np.max(coordinates[0]), np.max(coordinates[1]), np.max(coordinates[2])]))
    ##################### XY #####################
    mass_xy = []
    radius_xy = []
    k = 0
    for i in np.linspace(0, 2 * maximum, maxi):
        if i == 0:
            mass_xy.append(0)
            radius_xy.append(0)
        else:
            j = i
            distance_to_center_xy = ellipse_equation(coordinates[0], coordinates[1], angle_1, j, j * xy_b / xy_a)
            mass_xy.append(np.sum(masses[distance_to_center_xy < 1]))
            radius_xy.append(j)
            if k > 4:
                mean_3 = np.mean([mass_xy[k - 4], mass_xy[k - 3], mass_xy[k - 2]])
                if abs(mass_xy[k - 1] - mean_3) <= 1 * np.std([mass_xy[k - 4], mass_xy[k - 3], mass_xy[k - 2]]):
                    break
        k = k + 1
    if mass_xy[-1] < 1e2:
        h50_radius_xy, h20_radius_xy, h80_radius_xy, concentration_xy = np.NaN, np.NaN, np.NaN, np.NaN
    else:
        h50_radius_xy, _ = interpolated_intercept(np.array(radius_xy), np.array(mass_xy) / mass_xy[-1],
                                                  np.linspace(0.5, 0.5, len(mass_xy)))
        h80_radius_xy, _ = interpolated_intercept(np.array(radius_xy), np.array(mass_xy) / mass_xy[-1],
                                                  np.linspace(0.8, 0.8, len(mass_xy)))
        h20_radius_xy, _ = interpolated_intercept(np.array(radius_xy), np.array(mass_xy) / mass_xy[-1],
                                                  np.linspace(0.2, 0.2, len(mass_xy)))
        concentration_xy = 5 * np.log10(h80_radius_xy / h20_radius_xy)

    ##################### YZ #####################
    mass_yz = []
    radius_yz = []
    k = 0
    for i in np.linspace(0, 2 * maximum, maxi):
        if i == 0:
            mass_yz.append(0)
            radius_yz.append(0)
        else:
            j = i
            distance_to_center_yz = ellipse_equation(coordinates[1], coordinates[2], angle_2, j, j * yz_c / yz_b)
            mass_yz.append(np.sum(masses[distance_to_center_yz < 1]))
            radius_yz.append(j)
            if k > 4:
                mean_3 = np.mean([mass_yz[k - 4], mass_yz[k - 3], mass_yz[k - 2]])
                if abs(mass_yz[k - 1] - mean_3) <= 1 * np.std([mass_yz[k - 4], mass_yz[k - 3], mass_yz[k - 2]]):
                    break
        k = k + 1
    if mass_yz[-1] < 1e2:
        h50_radius_yz, h20_radius_yz, h80_radius_yz, concentration_yz = np.NaN, np.NaN, np.NaN, np.NaN
    else:
        h50_radius_yz, _ = interpolated_intercept(np.array(radius_yz), np.array(mass_yz) / mass_yz[-1],
                                                  np.linspace(0.5, 0.5, len(mass_yz)))
        h80_radius_yz, _ = interpolated_intercept(np.array(radius_yz), np.array(mass_yz) / mass_yz[-1],
                                                  np.linspace(0.8, 0.8, len(mass_yz)))
        h20_radius_yz, _ = interpolated_intercept(np.array(radius_yz), np.array(mass_yz) / mass_yz[-1],
                                                  np.linspace(0.2, 0.2, len(mass_yz)))
        concentration_yz = 5 * np.log10(h80_radius_yz / h20_radius_yz)

    ##################### XZ #####################
    mass_xz = []
    radius_xz = []
    k = 0
    for i in np.linspace(0, 2 * maximum, maxi):
        if i == 0:
            mass_xz.append(0)
            radius_xz.append(0)
        else:
            j = i
            distance_to_center_xz = ellipse_equation(coordinates[0], coordinates[2], angle_3, j, j * xz_c / xz_a)
            mass_xz.append(np.sum(masses[distance_to_center_xz < 1]))
            radius_xz.append(j)
            if k > 4:
                mean_3 = np.mean([mass_xz[k - 4], mass_xz[k - 3], mass_xz[k - 2]])
                if abs(mass_xz[k - 1] - mean_3) <= 1 * np.std([mass_xz[k - 4], mass_xz[k - 3], mass_xz[k - 2]]):
                    break
        k = k + 1
    if mass_xz[-1] < 1e2:
        h50_radius_xz, h20_radius_xz, h80_radius_xz, concentration_xz = np.NaN, np.NaN, np.NaN, np.NaN
    else:
        h50_radius_xz, _ = interpolated_intercept(np.array(radius_xz), np.array(mass_xz) / mass_xz[-1],
                                                  np.linspace(0.5, 0.5, len(mass_xz)))
        h80_radius_xz, _ = interpolated_intercept(np.array(radius_xz), np.array(mass_xz) / mass_xz[-1],
                                                  np.linspace(0.8, 0.8, len(mass_xz)))
        h20_radius_xz, _ = interpolated_intercept(np.array(radius_xz), np.array(mass_xz) / mass_xz[-1],
                                                  np.linspace(0.2, 0.2, len(mass_xz)))
        concentration_xz = 5 * np.log10(h80_radius_xz / h20_radius_xz)

    return (
        h50_radius_xy, h20_radius_xy, h80_radius_xy, concentration_xy,
        h50_radius_yz, h20_radius_yz, h80_radius_yz, concentration_yz,
        h50_radius_xz, h20_radius_xz, h80_radius_xz, concentration_xz
    )


def interpolated_intercept(x, y1, y2):
    """
    Computes the interpolated intercept of two curves.

    Args:
        x (array-like): Array of x-coordinates.
        y1 (array-like): Array of y-coordinates for the first curve.
        y2 (array-like): Array of y-coordinates for the second curve.

    Returns:
        tuple: A tuple containing the x and y coordinates of the interpolated intercept.
    """

    def line(p1, p2):
        a = p1[1] - p2[1]
        b = p2[0] - p1[0]
        c = p1[0] * p2[1] - p2[0] * p1[1]
        return a, b, -c

    def intersection(l1, l2):
        d = l1[0] * l2[1] - l1[1] * l2[0]
        dx = l1[2] * l2[1] - l1[1] * l2[2]
        dy = l1[0] * l2[2] - l1[2] * l2[0]
        return dx / d, dy / d

    idx = np.argwhere(np.diff(np.sign(y1 - y2)) != 0).flatten()

    point1 = [x[idx], y1[idx]]
    point2 = [x[idx + 1], y1[idx + 1]]
    point3 = [x[idx], y2[idx]]
    point4 = [x[idx + 1], y2[idx + 1]]

    xc, yc = intersection(line(point1, point2), line(point3, point4))

    return xc, yc


def solving_the_box_problem(dx, box_max):
    box_size = box_max
    if (len(dx[dx < 0.1 * box_size]) != 0) & (len(dx[dx > 0.9 * box_size]) != 0):
        half = 0.5 * box_size
        dx = np.where(dx > half, dx - box_size, dx)
    else:
        dx = dx
    return dx
