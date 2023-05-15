# This file is part of MOGPy: Python package for the study of galaxy morphology in universe simulations.
#
# MOGPy is free software: redistribution and use in source and binary forms,
# with or without modification, are permitted.

import h5py
import numpy as np
import pandas as pd
import astropy.units as u
import eagleSqlTools as sql
import os
import math
from astropy.table import Table

def read_dm_mass(Catalog_Ref, snapshot_name, file_name, cgs_M, a, aexp_M, h, hexp_M):
    """
    Read dark matter mass value and convert the unit to Solar Mass.

    Parameters:
    - Catalog_Ref (str): Path to the catalog reference.
    - snapshot_name (str): Name of the snapshot.
    - file_name (str): Name of the file.
    - cgs_M (float): Conversion factor from cgs to Solar Mass.
    - a (float): Scale factor.
    - aexp_M (float): Exponent for the scale factor in the mass conversion.
    - h (float): Hubble parameter.
    - hexp_M (float): Exponent for the Hubble parameter in the mass conversion.

    Returns:
    - m (float): Dark matter mass in Solar Mass units.
    """
    # Read dark matter mass value
    f = h5py.File(f'../data/catalogues_hdf5/{Catalog_Ref}/{snapshot_name}/{file_name}.0.hdf5', 'r')
    dm_mass = f['Header'].attrs.get('MassTable')[1]
    f.close()

    # Convert the unit to Solar Mass
    m = np.multiply(dm_mass, cgs_M * a**aexp_M * h**hexp_M * u.g.to(u.Msun), dtype='f8')

    return m

def max_groupnumber_sql(Catalog_Ref, snap_num, mass_min, mass_max):
    """
    Retrieve the maximum group number from the SQL database based on specified criteria.

    Parameters:
    - Catalog_Ref (str): Path to the catalog reference.
    - snap_num (int): Snapshot number.
    - mass_min (int): Minimum mass value.
    - mass_max (int): Maximum mass value.

    Returns:
    - mm (int): Maximum group number from the SQL database.
    """
    mySims = np.array([(Catalog_Ref, mass_max, snap_num)])
    print('You need your own access to database')
    con = sql.connect("", password="")

    for sim_name, mass_max, snap_num in mySims:
        myQuery = "SELECT SH.GroupNumber, SH.SubGroupNumber, SH.Snapnum, SH.Mass as total_mass \
                   FROM %s_SubHalo as SH \
                   WHERE SH.Mass > 1E%s and SH.Mass < 1E%s and SH.Snapnum = %s" % (sim_name, mass_min, mass_max, snap_num)
        myData = sql.execute_query(con, myQuery)

        try:
            mm = np.max(myData['GroupNumber'])
        except:
            mm = 0

        return mm
    
def CentreOfMass_sql(Catalog_Ref, snap_num, mass_max, gn):
    """
    Retrieve the center of mass data from the SQL database based on specified criteria.

    Parameters:
    - Catalog_Ref (str): Path to the catalog reference.
    - snap_num (int): Snapshot number.
    - mass_max (int): Maximum mass value.
    - gn (int): Group number.

    Returns:
    - myData (pandas.DataFrame): DataFrame containing the retrieved center of mass data.
    """
    mySims = np.array([(Catalog_Ref, mass_max, snap_num)])
    print('You need your own access to database')
    con = sql.connect("", password="")
 
    for sim_name, mass_max, snap_num in mySims:
        myQuery = "SELECT SH.GroupNumber, SH.SubGroupNumber, SH.Snapnum, SH.Mass as total_mass, \
                   SH.CentreOfMass_x, SH.CentreOfMass_y, SH.CentreOfMass_z \
                   FROM %s_SubHalo as SH \
                   WHERE SH.Mass > 1E%s and SH.Snapnum = %s and SH.GroupNumber = %s" % (sim_name, mass_max, snap_num, gn)
        myData = sql.execute_query(con, myQuery)
        myData = pd.DataFrame(myData)
        
        return myData

def read_eagle(Catalog_Ref, file_name, snapshot_name, snap_num, mass_min, mass_max):
    """
    Read EAGLE data from HDF5 files, process it, and save as FITS files.

    Parameters:
    - Catalog_Ref (str): Path to the catalog reference.
    - snapshot_name (str): Name of the snapshot.
    - snap_num (int): Snapshot number.
    - mass_min (float): Minimum mass value.
    - mass_max (float): Maximum mass value.

    Returns:
    - num (int): The number of iterations performed.
    """
    f = h5py.File(f'../data/catalogues_hdf5/{Catalog_Ref}/{snapshot_name}/{file_name}.0.hdf5', 'r')
    a = f['Header'].attrs.get('Time')
    h = f['Header'].attrs.get('HubbleParam')
    cgs_M  = f[f'PartType0/Mass'].attrs.get('CGSConversionFactor')
    aexp_M = f[f'PartType0/Mass'].attrs.get('aexp-scale-exponent')
    hexp_M = f[f'PartType0/Mass'].attrs.get('h-scale-exponent')
    cgs_C  = f[f'PartType0/Coordinates'].attrs.get('CGSConversionFactor')
    aexp_C = f[f'PartType0/Coordinates'].attrs.get('aexp-scale-exponent')
    hexp_C = f[f'PartType0/Coordinates'].attrs.get('h-scale-exponent')
    f.close()
    mass_dm = read_dm_mass(Catalog_Ref, snapshot_name, file_name, cgs_M, a, aexp_M, h, hexp_M)
    n_files = len([name for name in os.listdir(f'../data/catalogues_hdf5/{Catalog_Ref}/{snapshot_name}/') if os.path.isfile(os.path.join(f'../data/catalogues_hdf5/{Catalog_Ref}/{snapshot_name}/', name))])
    print("Check if Stage 1 is already done...")
    if os.path.exists(f'temp_{n_files-1}.fits'):
        pass
    else:
        for num in range(len([name for name in os.listdir(f'../data/catalogues_hdf5/{Catalog_Ref}/{snapshot_name}/') if os.path.isfile(os.path.join(f'../data/catalogues_hdf5/{Catalog_Ref}/{snapshot_name}/', name))])):
            result = pd.DataFrame([], columns = ['GroupNumber','SubGroupNumber','itype','Mass','Coordinates_x','Coordinates_y','Coordinates_z','Velocity_x','Velocity_y','Velocity_z','x','y','z'])
            data = {}
            for k in [0,1,4,5]:
                if k==0:
                    i=0
                else:
                    i=i+1
                
                data[i] = pd.DataFrame(np.array(h5py.File(f'../data/catalogues_hdf5/{Catalog_Ref}/{snapshot_name}/{file_name}.{num}.hdf5')[f'PartType{k}/GroupNumber']), columns = ['GroupNumber'])
                data[i]['SubGroupNumber'] = np.array(h5py.File(f'../data/catalogues_hdf5/{Catalog_Ref}/{snapshot_name}/{file_name}.{num}.hdf5')[f'PartType{k}/SubGroupNumber'])
                data[i]['itype'] = np.linspace(k,k,len(data[i]['GroupNumber']))
                if k == 1:
                    mass = pd.DataFrame(np.array(np.linspace(mass_dm,mass_dm,len(data[i]['GroupNumber']))), columns = ['Mass'])
                else:
                    mass = pd.DataFrame(np.array(h5py.File(f'../data/catalogues_hdf5/{Catalog_Ref}/{snapshot_name}/{file_name}.{num}.hdf5')[f'PartType{k}/Mass'])*cgs_M * a**aexp_M * h**hexp_M * u.g.to(u.Msun), columns = ['Mass'])
                if k == 0:
                    try:
                        tab = pd.DataFrame(np.array(h5py.File(f'../data/catalogues_hdf5/{Catalog_Ref}/{snapshot_name}/{file_name}.{num}.hdf5')[f'PartType{k}/Coordinates'])*cgs_C * a**aexp_C * h**hexp_C * u.cm.to(u.Mpc), columns = ['Coordinates_x','Coordinates_y','Coordinates_z'])
                    except:
                        continue
                tab = pd.DataFrame(np.array(h5py.File(f'../data/catalogues_hdf5/{Catalog_Ref}/{snapshot_name}/{file_name}.{num}.hdf5')[f'PartType{k}/Coordinates'])*cgs_C * a**aexp_C * h**hexp_C * u.cm.to(u.Mpc), columns = ['Coordinates_x','Coordinates_y','Coordinates_z'])
                vel = pd.DataFrame(np.array(h5py.File(f'../data/catalogues_hdf5/{Catalog_Ref}/{snapshot_name}/{file_name}.{num}.hdf5')[f'PartType{k}/Velocity']), columns = ['Velocity_x','Velocity_y','Velocity_z'])
                data[i] = data[i].join(mass)
                data[i] = data[i].join(tab)
                data[i] = data[i].join(vel)
                
                gn_max = max_groupnumber_sql(Catalog_Ref,snap_num,mass_min,mass_max)
                if math.isnan(gn_max):
                    gn_max = 0
                data[i] = data[i].loc[data[i]['GroupNumber']<gn_max]
            
            result = pd.concat([data[i] for i in range(len(data))])
            table = Table.from_pandas(result)
            table.write(f'temp_{num}.fits', format='fits', overwrite=True)
            print('File Number', num, ' done')
    print('Stage 1 done!')
    _ = merge_temp(Catalog_Ref, file_name, snapshot_name, snap_num, mass_min, mass_max)
    for num in range(len([name for name in os.listdir(f'../data/catalogues_hdf5/{Catalog_Ref}/{snapshot_name}/') if os.path.isfile(os.path.join(f'../data/catalogues_hdf5/{Catalog_Ref}/{snapshot_name}/', name))])):
        os.remove(f'temp_{num}.fits')
    print("Done!")
    return num

def merge_temp(Catalog_Ref, file_name, snapshot_name, snap_num, mass_min, mass_max):
    """
    Merge temporary FITS files and save the merged data.

    Returns:
    int: The number of iterations (i).
    """
    i=0
    temp_data = {}
    for num in range(len([name for name in os.listdir(f'../data/catalogues_hdf5/{Catalog_Ref}/{snapshot_name}/') if os.path.isfile(os.path.join(f'../data/catalogues_hdf5/{Catalog_Ref}/{snapshot_name}/', name))])):
        ddu = Table.read(f'temp_{num}.fits',format='fits')
        temp_data[num] = ddu.to_pandas()
    result = pd.concat([temp_data[i] for i in range(len([name for name in os.listdir(f'../data/catalogues_hdf5/{Catalog_Ref}/{snapshot_name}/') if os.path.isfile(os.path.join(f'../data/catalogues_hdf5/{Catalog_Ref}/{snapshot_name}/', name))]))])
    print(f"Saving")
    table = Table.from_pandas(result)
    table.write(f'../data/catalogues_fits/{Catalog_Ref}/{snapshot_name}/EAGLE_{Catalog_Ref}_Snapshot{snap_num}_MassMIN{mass_min}_MassMAX{mass_max}.fits', format='fits', overwrite=True)
    return i
