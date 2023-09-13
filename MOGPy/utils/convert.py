# This file is part of MOGPy: Python package for the study of galaxy morphology in universe simulations.
#
# MOGPy is free software: redistribution and use in source and binary forms,
# with or without modification, are permitted.

import pandas as pd
from astropy.table import Table


def csv_to_fits(csv_file, fits_file):
    """
    Convert a CSV file to a FITS file.

    Args:
    csv_file (str): Path to the input CSV file.
    fits_file (str): Path to save the output FITS file.

    Returns:
    None
    """
    # Read CSV file into a pandas DataFrame
    data = pd.read_csv(csv_file)

    # Convert DataFrame to an astropy Table
    table = Table.from_pandas(data)

    # Write the Table to a FITS file
    table.write(fits_file, format='fits', overwrite=True)
