# -*- coding: utf-8 -*-
#
# This scripts reads a netCDF tide model and extends the empty cells using
# the mean of 8 surrounding tide values. Afterwards, GMT grdlandmask is used
# to determine the water/land faction in each cell and scales the u and v
# values accordingly. The result is written to a new netCDF file.
#
# This script is part of otl_map 0.0.7
#
# 21/8/2023 Coimbra, Santa Clara
#
# (c) Copyright 2023 TeroMovigo, all rights reserved.
#===============================================================================

import math
import numpy as np
import os
import sys
import json
import argparse
from tqdm import tqdm
import netCDF4 as nc
import datetime
from otl_map.tidemodel cimport get_tidemodel_mesh, read_tidemodel
from otl_map.tidemodel cimport write_tidemodel

#===============================================================================
# Subroutines
#===============================================================================

cdef get_neighbours_mean(i,j,nx,ny,u,v):
    """ Get the u,v tide values for cell at i,j using 8 surrounding values

    Args:
        fname (string): name of file to read

    """

    cdef double    sum_u,sum_v,u_new,v_new
    cdef int       k,l,i0,j0,k0

    k = nx*j+i

    #--- For land cells, see if we can fill it
    if u[k]>999.0:
        sum_u = sum_v = 0.0
        l = 0
        for i0 in range(i-1,i+2):
            if i0<0:
                i0 += nx
            elif i0>=nx:
                i0 -= nx
            for j0 in range(j-1,j+2):
                if j0>=0 and j0<ny:
                    k0 = nx*j0 + i0
                    if u[k0]<999.0:
                        sum_u += u[k0]
                        sum_v += v[k0]
                        l     += 1

        #--- If we found water cells, fill central cell with mean
        if l>0:
            u_new = sum_u/l
            v_new = sum_v/l
        #--- Otherwise, simply land
        else:
            u_new = 999.9
            v_new = 999.9
    #--- Water stays water
    else:
        u_new = u[k]
        v_new = v[k]

    return [u_new,v_new]



cdef read_mask(tidemodel):
    """

    """

    #--- Find file
    # water fractions of `otl-map masks` (otl_map.paths.masks_dir)
    from otl_map.paths import masks_dir
    cpp_dir = str(masks_dir())
    fname = os.path.join(cpp_dir, 'mask_' + tidemodel + '.nc')

    #--- Read amplitude and phase-lag map
    try:
        nc_mask = nc.Dataset(fname)
    except:
        print('mask: could not read {0:s}'.format(fname))
        sys.exit()

    #--- Get size
    dim = nc_mask.variables['dimension']
    nx = dim[0]
    ny = dim[1]

    return nc_mask.variables['z'][:]


#===============================================================================
# Main program
#===============================================================================


def main():

    cdef int       i,j,nx,ny
    cdef double    rad,x0,y1,sp0,sp1
    cdef double[:] xr
    cdef double[:] yr
    cdef double[:] sp
    cdef double[:] u_new
    cdef double[:] v_new

    #--- Constants
    rad = math.pi/180.0

    #--- Parse command line arguments in a bit more professional way
    parser = argparse.ArgumentParser(description="Modify tide model")

    #--- List arguments that can be given 
    parser.add_argument('-i', dest='fname', action='store', required=False,
           default="modify_tidemodel.json", help="The name of the json file")

    args = parser.parse_args()

    #--- Store the name of the input and output file
    fname  = args.fname


    #--- Read control parameters
    try:
        fp = open(fname,'r')
        configuration = json.load(fp)
        fp.close()
    except:
        check_file = os.path.isfile(fname)
        if check_file==False:
            print('Could not find file: {0:s}'.format(fname))
        else:
            print('Could not parse: {0:s}'.format(fname))
            sys.exit()

    try:
        tidemodel_in = configuration['tidemodel_in']
    except:
        print('Could not find tidemodel_in keyword')
        sys.exit()
    try:
        tidemodel_out = configuration['tidemodel_out']
    except:
        print('Could not find tidemodel_out keyword')
        sys.exit()
    try:
        harm = configuration['harmonic']
    except:
        harm = 'm2'
    try:
        extend = bool(configuration['extend'])
    except:
        print('Could not find extend keyword')
        sys.exit()
    try:
        verbose = bool(configuration['verbose'])
    except:
        verbose = True

    #--- Get Tide Model mesh info
    [nx,ny,x0,y1,sp0,sp1] = get_tidemodel_mesh(tidemodel_in,harm,verbose)

    if verbose==True:
        print('\nModify Tide model')
        print('-----------------')
        print('tidemodel_in  : {0:s}'.format(tidemodel_in))
        print('tidemodel_out : {0:s}'.format(tidemodel_out))
        print('extend        : {0:}'.format(extend))

    #--- Fill standard arrays
    xr = np.zeros(2)
    yr = np.zeros(2)
    sp = np.zeros(2)
    xr[0] = x0
    xr[1] = x0+nx*sp0
    yr[0] = y1-ny*sp1
    yr[1] = y1
    sp[0] = sp0
    sp[1] = sp1

    #--- Allocate some memory to various arrays
    u     = np.zeros(nx*ny, dtype=np.double)
    v     = np.zeros(nx*ny, dtype=np.double)
    u_new = np.zeros(nx*ny, dtype=np.double)
    v_new = np.zeros(nx*ny, dtype=np.double)
    if extend==True:
        u_dummy = np.zeros(nx*ny, dtype=np.double)
        v_dummy = np.zeros(nx*ny, dtype=np.double)

    read_tidemodel(tidemodel_in,harm,u,v,999.9,verbose)

    #--- Do we need to extend the model?
    if extend==True:
        for i in range(0,nx):
            for j in range(0,ny):
                k = nx*j+i
                [u_dummy[k],v_dummy[k]] = get_neighbours_mean(i,j,nx,ny,u,v)
        u = u_dummy
        v = v_dummy

    #--- Modify the model
    mask = read_mask(tidemodel_in)
    for k in range(0,nx*ny):
        if u[k]<999.0:
            u_new[k] = mask[k]*u[k]
            v_new[k] = mask[k]*v[k]
        else:
            u_new[k] = 999.9
            v_new[k] = 999.9

    #--- Done, write results to file
    write_tidemodel(tidemodel_out,harm,u_new,v_new,xr,yr,sp,nx,ny)
