# MOGPy (Morphology Of Galaxies in Python)

![PyPi python support](https://img.shields.io/badge/Python-3.8-blue)

MOGPy is a custom Python package designed to provide a comprehensive set of tools and methods for estimating the shape, orientation, and mass distribution of galaxies based on data from universe simulations such as EAGLE or IllustrisTNG. It offers a wide range of functionalities for analyzing and characterizing the morphology of galaxies, enabling researchers to gain deeper insights into the structure and properties of simulated galaxies.

## /!\ MOGPy package is not yet fully released.

You can find my first article in A&A using MOGPy: [Shape, alignment, and mass distribution of baryonic and dark-matter halos in one EAGLE simulation](https://www.aanda.org/articles/aa/full_html/2023/01/aa44920-22/aa44920-22.html)

Key Features:

1. Shape Estimation: MOGPy provides advanced methods for estimating the shapes of galaxies within the simulation data. It incorporates techniques such as moment of inertia, principal component analysis, and elliptical fitting to accurately determine the shape parameters of galaxies.
2. Orientation Determination: The package offers tools to determine the orientation of galaxies in the simulated universe. By analyzing the spatial distribution and alignment of galaxy components, MOGPy enables to understand the intrinsic orientation of galaxies within the simulated environment.
3. Mass Distribution Analysis: MOGPy includes capabilities for analyzing the mass distribution within galaxies. It enables the extraction of mass profiles, density profiles, and concentration measures, providing valuable insights into the mass distribution and concentration of galaxies in the simulated universe.
4. Visualization and Reporting: MOGPy offers visualization tools to generate informative plots, diagrams, and visual representations of galaxy morphology and mass distribution. It enables researchers to create publication-quality figures and reports summarizing their analysis results.

## Installation

To install MOGPy, follow these steps:

1. Clone the repository to your local machine using `git clone https://github.com/Etthj/MOGPy`.
2. Navigate to the repository directory.

Good practice is to create a new python environment:
```sh
conda create -n Mogpy-env python=3.7
conda activate Mogpy-env
```

### External package requirements

- `astropy` (5.0.1)
- `eagleSqlTools` (2.0.0)
- `h5py` (3.6.0)
- `numpy` (1.22.2)
- `pandas` (1.4.1)
- `scipy` (1.8.0)
- `sep` (1.2.0)
- `sql` (2022.4.0)
- `wget` (3.2)

## Usage

The repository should contain the following files and directories:

1. MOGPy/: This directory contains the source code and implementation of the MOGPy package.
  - main.py: The main module of MOGPy containing methods for estimating the shapes of galaxies.
  - utils/functions.py: The module of MOGPy containing the core functionalities and methods.
  - utils/eagle_functions.py: EAGLE simulations related functions.
2. examples/: This directory holds example scripts and notebooks demonstrating the usage of MOGPy for various tasks.
3. data/: This directory contains sample datasets or simulation outputs that can be used for testing and running the examples.
4. LICENSE: The license file specifying the terms and conditions for using the MOGPy package.
5. README.md: The README file you are currently reading, providing an overview of the repository and its contents.

## Example notebooks

Below you can find a few examples that can be tackled with `MOGPy`:
- [**Starting guide**: Estimation of galaxy shape parameters using MOGPy](https://github.com/Etthj/MOGPy/blob/main/examples/shape_estimation_EAGLE_data.ipynb)
- [Downloading and viewing EAGLE simulations dataset](https://github.com/Etthj/MOGPy/blob/main/examples/download_and_visualise_EAGLE_data.ipynb)

We hope that this tutorial will serve as a valuable resource for learning how to analyze universe simulation data and gain insights into the morphology of galaxies.

**Note:** If you encounter any issues or have questions, please feel free to open an issue in the GitHub repository.

## License

MOGPy is distributed under the *MIT License*. Feel free to use, modify, and distribute the package for personal or commercial purposes, but please include the appropriate attribution and citation if you use this work in your research or publications.

## Contact

For further inquiries or assistance, you can contact the repository owner at quentin.petit.etthj@proton.me.

Enjoy exploring the fascinating world of galaxy morphology analysis with MOGPy.
