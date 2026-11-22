# -*- coding: utf-8 -*-
#
# This file simply provides two subroutines to get the dimensions of the
# ocean tide grid and another one to read the amplitude and phase-lag 
# values into arrays.
#
# This script is part of otl_map 0.0.7
#
# 18/3/2023 Coimbra, Santa Clara
#
# (c) Copyright 2023 TeroMovigo, all rights reserved.
#===============================================================================

import os
from otl_map.paths import derived_dir, greenfnc_dir, tidemodel_dir
import sys
import json
import netCDF4 as nc
import numpy as np
from libc.math cimport sin,cos,acos,sqrt,atan2,fabs,pi,asin

#===============================================================================
# Class definition
#===============================================================================


cdef get_tidemodel_mesh(tidemodel,harm,verbose):
    """ Get information about array sizes in netCDF file 

    """

    cdef int    nx,ny
    cdef double xr[2]
    cdef double yr[2]
    cdef double sp[2]
   
    #--- Constants 
    cdef rad = pi/180.0

    #--- Create directory name with amp and pha files
    directory = tidemodel_dir(tidemodel) + '/'

    #--- Read amplitude and phase-lag map
    try:
        nc_amp = nc.Dataset(directory + 'amp.' + harm)
    except:
        print('{0:1}: could not read amp.{1:s}'.format(tidemodel,harm))
        sys.exit()
    try:
        nc_pha = nc.Dataset(directory + 'pha.' + harm)
    except:
        print('{0:1}: could not read pha.{1:s}'.format(tidemodel,harm))
        sys.exit()

    #--- Compute u and v map
    dim = nc_amp.variables['dimension']
    nx = dim[0]
    ny = dim[1]

    #--- Get grid info
    xr = nc_amp.variables['x_range'][:]
    yr = nc_amp.variables['y_range'][:]
    sp = nc_amp.variables['spacing'][:]

    #--- close NetCDF's
    nc_amp.close()
    nc_pha.close()

    #--- Some grids (e.g. FES2022b) include both 0° and 360° as separate columns,
    #    making nx one too many for a periodic 360° domain.  Detect and drop the
    #    duplicate so downstream FFT code sees a clean cnx*csp0 == 2π grid.
    if fabs((nx-1)*sp[0] - 360.0) < 1e-4:
        nx -= 1

    if (verbose==True):
        print("\nTide model info\n-----------------")
        print("tidemodel : {0:s}".format(tidemodel))
        print('nx = {0:d}'.format(nx))
        print('ny = {0:d}'.format(ny))
        print('xr=',xr)

    return [nx,ny,rad*xr[0],rad*yr[1],rad*sp[0],rad*sp[1]]



cdef read_tidemodel(tidemodel,harm,u,v,land_value,verbose):
    """ Read netCDF file into arrays u and v

    Args:
        tidemodel (string)  : name of tide model
        harm (string)       : harmonic (m2, s2, ...)
        u,v (array double)  : real and quadrature tide amplitudes [output]
        land_value (double) : value to put over land in the arrays
        verbose (bool)      : print some stuff on screen
    """

    cdef int       i,j,nx,ny,nx_file
    cdef float[:]  a
    cdef float[:]  p

    #--- Constants
    cdef rad = pi/180.0

    #--- Create directory name with amp and pha files
    directory = tidemodel_dir(tidemodel) + '/'

    #--- Read amplitude and phase-lag map
    try:
        nc_amp = nc.Dataset(directory + 'amp.' + harm)
    except:
        print('{0:1}: could not read amp.{1:s}'.format(tidemodel,harm))
        sys.exit()
    try:
        nc_pha = nc.Dataset(directory + 'pha.' + harm)
    except:
        print('{0:1}: could not read pha.{1:s}'.format(tidemodel,harm))
        sys.exit()

    #--- Compute u and v map
    dim = nc_amp.variables['dimension']
    nx_file = dim[0]
    ny      = dim[1]

    #--- Detect wrapped grid (e.g. FES2022b includes duplicate column at 360°=0°).
    sp_deg = float(nc_amp.variables['spacing'][0])
    nx = nx_file
    if fabs((nx_file - 1) * sp_deg - 360.0) < 1e-4:
        nx = nx_file - 1

    a = np.zeros(nx_file*ny, dtype=np.float32)
    p = np.zeros(nx_file*ny, dtype=np.float32)
    try:
        a = nc_amp.variables['z'][:]
        p = nc_pha.variables['z'][:]
    except Exception as error:
        print("An exception occurred:", error)
        print('Trying to proceed with doubles instead of floats...')

        aa = np.zeros(nx_file*ny)
        pp = np.zeros(nx_file*ny)
        aa = nc_amp.variables['z'][:]
        pp = nc_pha.variables['z'][:]
        for i in range(0, nx_file*ny):
            a[i] = <float>aa[i]
            p[i] = <float>pp[i]

        print('survived this trick...')

    #--- close NetCDF's
    nc_amp.close()
    nc_pha.close()

    #--- Fill output arrays; skip the duplicate last column if nx < nx_file.
    for j in range(0, ny):
        for i in range(0, nx):
            k_file = nx_file*j + i
            k_out  = nx*j + i
            if a[k_file] < 999.0:
                u[k_out] = a[k_file] * cos(rad*p[k_file])
                v[k_out] = a[k_file] * sin(rad*p[k_file])
            else:
                u[k_out] = land_value
                v[k_out] = land_value

    if (verbose==True):
        print("\nTide model\n-----------------")
        print("harmonic  : {0:s}".format(harm))
        print("land_value: {0:f}".format(land_value))



cdef write_tidemodel(tidemodel,harm,u,v,xr,yr,sp,nx,ny):
    """ Write tidemodel to a netCDF file

    Args:
        tidemodel (string)  : name of tide model
        harm (string)       : harmonic (m2, s2, ...)
        u,v (array double)  : real and quadrature tide amplitudes [input]
        xr (array double[2]): left and right boundary of grid
        yr (array double[2]): lower and upper boundary of grid
        sp (array double[2]): x and y grid spaceing
        nx,ny (int)         : number of cells in x and y direction
    """

    cdef int       i,j
    cdef double    a_min,a_max,p_min,p_max
    cdef double[:]  amp
    cdef double[:]  pha
   
    #--- Constants 
    cdef rad = pi/180.0

    #--- Create directory name with amp and pha files
    directory = derived_dir(tidemodel) + '/'
    if not os.path.exists(directory):
       os.makedirs(directory)


    #--- Convert back to degrees
    for i in range(0,2):
        xr[i] /= rad
        yr[i] /= rad
        sp[i] /= rad

    #--- Convert to amplitude and phase-lags
    amp = np.zeros(nx*ny)
    pha = np.zeros(nx*ny)
    for i in range(0,nx*ny):
        if u[i]<999.0:
           amp[i] = sqrt(u[i]*u[i] + v[i]*v[i])
           pha[i] = atan2(v[i],u[i])/rad
           if i==0:
               a_min = a_max = amp[i]
               p_min = p_max = pha[i]
           else:
               if amp[i]<a_min:
                   a_min = amp[i]
               if amp[i]>a_max:
                   a_max = amp[i]
               if pha[i]<p_min:
                   p_min = pha[i]
               if pha[i]>p_max:
                   p_max = pha[i]

    #--- Create amplitude netCDF file
    try:
        nc_amp = nc.Dataset(directory + 'amp.' + harm,'w')
    except:
        print('{0:1}: could not create amp.{1:s}'.format(tidemodel,harm))
        sys.exit()
    side     = nc_amp.createDimension("side",2)
    xysize   = nc_amp.createDimension("xysize",nx*ny)
    dimension= nc_amp.createVariable("dimension","i4",("side",))
    x_range  = nc_amp.createVariable("x_range","f8",("side",))
    y_range  = nc_amp.createVariable("y_range","f8",("side",))
    z_range  = nc_amp.createVariable("z_range","f8",("side",))
    spacing  = nc_amp.createVariable("spacing","f8",("side",))
    z        = nc_amp.createVariable("z","f4",("xysize",))
    x_range.units = 'degrees'
    y_range.units = 'degrees'
    z_range.units = 'meters'
    z.long_name = 'amplitude'

    dimension[:] = [nx,ny]
    x_range[:]   = xr
    y_range[:]   = yr
    z_range[:]   = [a_min,a_max]
    spacing[:]   = sp
    z[:]         = amp
    nc_amp.close()

    #--- Create phase-lag netCDF file
    try:
        nc_pha = nc.Dataset(directory + 'pha.' + harm,'w')
    except:
        print('{0:1}: could not create amp.{1:s}'.format(tidemodel,harm))
        sys.exit()
    side     = nc_pha.createDimension("side",2)
    xysize   = nc_pha.createDimension("xysize",nx*ny)
    dimension= nc_pha.createVariable("dimension","i4",("side",))
    x_range  = nc_pha.createVariable("x_range","f8",("side",))
    y_range  = nc_pha.createVariable("y_range","f8",("side",))
    z_range  = nc_pha.createVariable("z_range","f8",("side",))
    spacing  = nc_pha.createVariable("spacing","f8",("side",))
    z        = nc_pha.createVariable("z","f4",("xysize",))
    x_range.units = 'degrees'
    y_range.units = 'degrees'
    z_range.units = 'degrees'
    z.long_name = 'phase-lag'

    dimension[:] = [nx,ny]
    x_range[:]   = xr
    y_range[:]   = yr
    z_range[:]   = [p_min,p_max]
    spacing[:]   = sp
    z[:]         = pha
    nc_pha.close()
