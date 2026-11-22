# -*- coding: utf-8 -*-
#
# This computes OTL maps using the brute force approach. For each mesh cell
# it simply computes the contribution of all surrounding ocean cells.
#
# This script is part of otl_map 0.0.7
#
# 15/3/2023 Coimbra, Santa Clara
#  6/8/2023 Coimbra, Santa Clara
#
# (c) Copyright 2023 TeroMovigo, all rights reserved.
#===============================================================================

import math
import numpy as np
import os
from otl_map.paths import greenfnc_dir, tidemodel_dir
import sys
import json
import netCDF4 as nc
import argparse
from otl_map.greensfunction import Greensfunction
from otl_map.geometry cimport compute_distance
from otl_map.cell_integration cimport integrate_cell,numerical_integral
from tqdm import tqdm
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
import datetime
from libc.math cimport sin,cos,acos,sqrt,atan2,fabs,pi,asin

#===============================================================================
# Subroutines
#===============================================================================

#--- Cython struct definition
cdef struct ocean_cell:
    double lon,lat,u,v



cdef read_tidemodel(tidemodel,harm,verbose):
    """ Read netCDF file into arrays u and v

    """

    cdef int    i,j,nx,ny
   
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
    u  = np.zeros(nx*ny)
    v  = np.zeros(nx*ny)

    #--- Get grid info
    xr = nc_amp.variables['x_range'][:]
    yr = nc_amp.variables['y_range'][:]
    sp = nc_amp.variables['spacing'][:]
     
    try:
        a = nc_amp.variables['z'][:]
        p = nc_pha.variables['z'][:]
    except:
        print('could not read amp and pha arrays')
        sys.exit()

    for i in range(0,nx*ny):
        if a[i]<999.0:
            u[i] = a[i] * cos(rad*p[i])
            v[i] = a[i] * sin(rad*p[i])

    #--- close NetCDF's
    nc_amp.close()
    nc_pha.close()

    if (verbose==True):
        print("\nTide model\n-----------------")
        print("harmonic  : {0:s}".format(harm))
        print("tidemodel : {0:s}".format(tidemodel))
        print('nx = {0:d}'.format(nx))
        print('ny = {0:d}'.format(ny))
        print('xr=',xr)

    return [u,v,nx,ny,rad*xr[0],rad*yr[1],rad*sp[0],rad*sp[1]] 


#===============================================================================
# Main program
#===============================================================================

def main():

    #--- Start clock
    start_time = datetime.datetime.now()

    #--- Constants 
    cdef rad = math.pi/180.0
    cdef R = 6.371e6
    cdef tpi = 2*math.pi

    #--- Variables
    cdef int        i,j,k,l,cnx,cny
    cdef double     x0,x1,y0,y1,sp0,sp1,rho
    cdef double     A,G,lon_c,lat_c,s_u,s_v
    cdef double     cx0,xy1,csp0,csp1,csm,snm,csc,snc
    cdef double     theta,alpha,yy0,yy1,scale,hpi
    cdef ocean_cell *mesh
    cdef double     *ptheta=&theta
    cdef double     *palpha=&alpha
    cdef double[:]  u
    cdef double[:]  v
    cdef double[:]  cc


    #--- Parse command line arguments in a bit more professional way
    parser = argparse.ArgumentParser(description="compute OTL map the hard way")

    #--- List arguments that can be given 
    parser.add_argument('-i', dest='fname', action='store', 
           default='otl_map.json', required=False, 
			help="The name of the json file with configuration")

    args = parser.parse_args()

    #--- Store the name of the input and output file
    fname  = args.fname

    #--- Array to hold coefficients for near analytical integration
    cc = np.zeros(4, dtype=np.double)

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
        tidemodel = configuration['tidemodel']
    except:
        print('Could not find tidemodel keyword')
        sys.exit()
    try:
        harms = configuration['harmonics']
    except:
        harms = ['m2']
    if len(harms)>1:
        print('Only the first harmonic will be computed!')
    harm = harms[0]
   
    try:
        [x0,x1,y0,y1,sp0,sp1] = configuration['mesh']
    except:
        print('Could not find tidemodel mesh')
        sys.exit()
    try:
        rho = float(configuration['rho'])
    except:
        rho = 1030.0
    try:
        verbose = bool(configuration['verbose'])
    except:
        verbose = True 
    try:
        label = configuration['label']
    except:
        label = 'brute'
    try:
        ccc = configuration['constants']
        for i in range(0,4):
            cc[i] = ccc[i]
    except:
        print('Cannot find constants approximating greensfnc')
        sys.exit()


    print('OTL map computation (slow)')
    print('--------------------------')
    print('mesh   : {0:f}, {1:f}, {2:f}, {3:f}, {4:f}, {5:f}'. \
					format(x0,x1,y0,y1,sp0,sp1))
    print('rho    : {0:f}'.format(rho))
    print('label  : {0:s}\n'.format(label))
    print('c[0]   : {0:e}'.format(cc[0]))
    print('c[1]   : {0:e}'.format(cc[1]))
    print('c[2]   : {0:e}'.format(cc[2]))
    print('c[3]   : {0:e}\n'.format(cc[3]))

    #--- Get Tide Model mesh info
    [u,v,cnx,cny,cx0,cy1,csp0,csp1] = read_tidemodel(tidemodel,harm,verbose)


    #--- Get Green's function
    greensfnc = Greensfunction(fname)

    #--- Count ocean cells
    n_cells = 0
    for k in range(cnx*cny):
        if u[k]<999.0:
            n_cells = n_cells + 1

    #--- Create array to store ocean cells
    cdef ocean_cell* ocean
    cdef ocean_cell  cell
    ocean = <ocean_cell*> PyMem_Malloc(n_cells*sizeof(ocean_cell))
     
    l = 0 
    for i in np.arange(0,cnx):
        lon = cx0 + (0.5 + i)*csp0
        for j in np.arange(0,cny):
            lat = cy1 - (0.5 + j)*csp1
            k = cnx*j + i
            if u[k]<999.0:
                cell.lon = lon
                cell.lat = lat
                cell.u   = u[k]
                cell.v   = v[k]
                ocean[l] = cell
                l += 1
                

    #--- Compute number of rows and columns
    EPS = 1.0e-8
    nx = int((x1-x0)/sp0 + EPS) + 1   
    ny = int((y1-y0)/sp1 + EPS) + 1   

    #--- Write result to file
    nc_amp = nc.Dataset('amp_{0:s}.nc'.format(label),'w')
    nc_pha = nc.Dataset('pha_{0:s}.nc'.format(label),'w')
    dx = nc_amp.createDimension('x',nx)
    dy = nc_amp.createDimension('y',ny)
    vx = nc_amp.createVariable('x', np.double, 'x')
    vy = nc_amp.createVariable('y', np.double, 'y')
    vx[:] = np.arange(x0,x1+sp0,sp0)
    vy[:] = np.arange(y0,y1+sp1,sp1)
    va = nc_amp.createVariable('z', 'f4', ('y','x'))
    dx = nc_pha.createDimension('x',nx)
    dy = nc_pha.createDimension('y',ny)
    vx = nc_pha.createVariable('x', np.double, 'x')
    vy = nc_pha.createVariable('y', np.double, 'y')
    vx[:] = np.arange(x0,x1+sp0,sp0)
    vy[:] = np.arange(y0,y1+sp1,sp1)
    vp = nc_pha.createVariable('z', 'f4', ('y','x'))

    #--- To avoid always calculating the following numbers, put it in 'scale'
    sp0 *= rad
    sp1 *= rad
    y0  *= rad
    x0  *= rad
    hpi  = rad*90.0
    scale= pow(R,2.0)*csp0

    #--- Loop over the list of grid cells for which OTL needs to be computed
    for i in tqdm(np.arange(0,nx)):
        lon_m = x0 + i*sp0
        for j in np.arange(0,ny):
            lat_m = y0 + j*sp1
            csm   = cos(lat_m) # compute_distance only uses cos and sin of lat
            snm   = sin(lat_m)
             
            #--- compute OTL
            s_u = 0.0
            s_v = 0.0

            #--- Loop over each ocean tide cell
            for k in range(n_cells):
                cell = ocean[k]

                #--- Area of cell
                yy0 = cell.lat - 0.5*csp1
                yy1 = cell.lat + 0.5*csp1
                if yy0<-hpi:
                    yy0 = -hpi
                elif yy1>hpi:  
                    yy1 = hpi
                A = scale*(sin(yy1)-sin(yy0))
                csc = cos(cell.lat)
                snc = sin(cell.lat)

                compute_distance(lon_m,csm,snm,cell.lon,csc,snc,ptheta,palpha)

                #--- Get value of Green's function
                if theta<3*csp0:
                    if lon_m-cell.lon>pi:
                        xp = R*csm*(lon_m - cell.lon - tpi)
                    elif lon_m-cell.lon<-pi:
                        xp = R*csm*(lon_m - cell.lon + tpi)
                    else:
                        xp = R*csm*(lon_m-cell.lon)
                    yp = R*(lat_m-cell.lat)
                    dx = R*csm*csp0
                    dy = R*csp1
                    if theta + csp0<0.15*rad:
                        G = rho*integrate_cell(greensfnc.component,\
                                                            cc,-xp,yp,dx,dy)
                    else:
                        G = rho*numerical_integral(greensfnc.component,\
                                                     greensfnc,-xp,yp,dx,dy)

                else:
                    #--- Get value of Green's function
                    G = greensfnc.interpolate(theta)

                    if greensfnc.component==1:
                        G *= cos(alpha-pi)
                    elif greensfnc.component==2:
                        G *= -sin(alpha-pi)

                #--- Add result to sum
                s_u += rho*G*A*cell.u
                s_v += rho*G*A*cell.v
  
            amp = math.sqrt(s_u*s_u + s_v*s_v)
            pha = math.atan2(s_v,s_u)/rad 
            va[j,i] = 1000.0*amp
            vp[j,i] = pha

    nc_amp.close()
    nc_pha.close()

    #--- Free memory 
    PyMem_Free(ocean)

    #--- Stop the clock
    end_time = datetime.datetime.now()
    print('\nTotal computation time : {0:}'.format(end_time - start_time))
