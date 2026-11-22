# -*- coding: utf-8 -*-
#
# This file contains the class "Greensfunction" for reading the Green's
# function into memory and provide an subroutine for getting its value
# at any given angular distance.
#
# This script is part of otl_map 0.0.7
#
#  9/3/2023 Coimbra, Santa Clara
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
cimport cython
from libc.math cimport fabs, fmin, log
cimport numpy as cnp


#===============================================================================
# Class definition
#===============================================================================


cdef class Greensfunction:
    ''' Class to read Green's function and provide interpolation subroutine
    '''

    def __init__(self,fname='otl_map.json', keyword='greensfunction'):
        ''' Read file and select correct component
        '''

        #--- Constants 
        self.rad = rad = math.pi/180.0
        component_name = ["vertical displacement",\
                               "horizontal displacement N-S",\
                               "horizontal displacement E-W"]

        #--- Initialise bracket of table
        self.il = self.iu = 0

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
            greensfunction = configuration[keyword] 
        except:
            print('Could not find {0:s} keyword'.format(keyword))
            sys.exit()
        try:
            self.component = configuration['component']
        except:
            print('Could not find component keyword')
            sys.exit()
        try:
            self.verbose = configuration['verbose']
        except:
            self.verbose = True

        #--- Read Green's function
        directory = greenfnc_dir()
        try:
            fp_in = open(directory + '/' + greensfunction,'r')
        except:
            print('Could not open {0:s}'.format(
					directory + '/' + greensfunction))
            sys.exit()
        line = fp_in.readline()
        if (self.verbose==True):
            print("\nRead Green's function\n---------------------")
            print("file      : {0:s}".format(greensfunction))
            print("header    : {0:s}".format(line.strip()))
            print("component : {0:s}".format(\
					component_name[self.component]))
 
        #--- With all the preliminary stuff out of the way, read file
        x_lst = []
        y_lst = []
        for line in fp_in:
            cols = line.split()
            x_lst.append(self.rad*float(cols[0]))
            if self.component==0:
                y_lst.append(float(cols[1]))
            elif self.component==1 or self.component==2:
                y_lst.append(float(cols[2]))
            else:
                print('Component {0:d} has not been implemented'.format(\
							   self.component))
                sys.exit()
        fp_in.close()

        #--- work with np.arrays
        self.n = len(x_lst)
        self.ln_x = np.zeros(self.n)
        self.x = np.array(x_lst)
        self.y = np.array(y_lst)
        for i in range(0,self.n):
            self.ln_x[i] = log(self.x[i])
						
        #--- Already apply scaling
        if self.component<3:
           for i in range(0,self.n):  
               self.y[i] = self.y[i] * self.x[i]
        else:
           print('Scaling not yet implemented for component {0:d}'.\
							format(self.component))
           sys.exit() 

        #--- Prepare for cubic splines
        self.initialise_spline(self.ln_x,self.y)



    cdef initialise_spline(self,double[:] x, double[:] y):
        """ Compute second derivative of Green's function. 

        Algorithm copied  from Numerical Recipes.
        """
  
        cdef int        i
        cdef double    yp1,ypn,un,sig,p,qn
        cdef double[:] u

        u = np.empty(self.n)
        self.y2 = np.empty(self.n)
        yp1  = 0.0
        ypn  = 0.0
        self.y2[0]= -0.5
        u[0] = (3.0/(x[1]-x[0]))*((y[1]-y[0])/(x[1]-x[0])-yp1)
        i = 1
        while i<self.n-1:
            sig=(x[i]-x[i-1])/(x[i+1]-x[i-1])
            p=sig*self.y2[i-1]+2.0
            self.y2[i]=(sig-1.0)/p
            u[i]=(y[i+1]-y[i])/(x[i+1]-x[i]) - (y[i]-y[i-1])/(x[i]-x[i-1])
            u[i]=(6.0*u[i]/(x[i+1]-x[i-1])-sig*u[i-1])/p
            i += 1
    
        qn = 0.5
        un = (3.0/(x[self.n-1]-x[self.n-2]))*(ypn- \
			(y[self.n-1]-y[self.n-2])/(x[self.n-1]-x[self.n-2]))
        self.y2[self.n-1] = (un-qn*u[self.n-2])/(qn*self.y2[self.n-2]+1.0)

        i=self.n-2
        while i>=0:
            self.y2[i] = self.y2[i]*self.y2[i+1]+u[i]
            i -= 1
  
   
 
    cpdef double spline(self, unsigned int il, double xx):
        """ Compute spline (taken from Numerical Recipes)
   
        Args:
            il (int): index of array just before the requested x-value.
            xx (double): x-value
 
        Returns interpolated y-value
        """
 
        cdef unsigned int    iu
        cdef double h,a,b

        iu = il+1
        h  = self.ln_x[iu]-self.ln_x[il]
        a  = (self.ln_x[iu]-xx)/h
        b  = (xx-self.ln_x[il])/h
        return a*self.y[il]+b*self.y[iu]+\
			((a*a*a-a)*self.y2[il]+(b*b*b-b)*self.y2[iu])*(h*h)/6.0
 
 

    cpdef double _interpolate(self,double xx):
        """ Bracket value in table and call spline

        Args:
            xx (double) : angular distance in radians

        Returns:
            interpolated value (double)
        """

        cdef unsigned int  i,incr

        #--- First call, so set search interval to first and last entry of table
        if self.il==0 and self.iu==0:
            self.il = 0
            self.iu = self.n - 1

        #--- slowly increase search interval to include value of phi 
        else:
            incr = 1
            while self.il>0 and self.ln_x[self.il]>xx:
                self.il -= incr
                incr *= 2
        
            if self.il<0:
                self.il=0
            incr = 1
            while self.iu<self.n and self.ln_x[self.iu]<xx:
                self.iu += incr
                incr *= 2
        
            if self.iu>=self.n:
                self.iu=self.n-1
      
        #--- Search interval brackets phi so start bi-secting!
        while self.iu-self.il>1: # Bisection to bracket correct interval
            i=(self.iu+self.il) >> 1
            if self.ln_x[i] < xx: 
                self.il=i 
            else:
                self.iu=i
      
        return self.spline(self.il,xx) 



    cpdef double interpolate(self, double xx):
        """ Interpolate the Green's function

        Args:
            xx (float): angular distance in radians

        Returns:
            yy (float): interpolated value, corrected for scale
        """

        cdef:
            double yy

        #--- Check range
        if xx<1.0e-6:
            return self.y[0]
        else:
            if xx<self.x[1] or xx>self.x[self.n-1]:
                print('angular distance {0:e} outside range!'.\
							format(xx/self.rad))
                sys.exit()

            #--- Use cubic spline interpolation
            yy = self._interpolate(log(xx))

            #--- Undo scaling
            if self.component<3:
                yy /= xx
            else:
                print('Scaling not yet implemented for component {0:d}'.\
							format(self.component))
                sys.exit()

            return yy


    def interpolate_array(self, double[:] xx_mv):
        """Vectorised Green's function interpolation over an array of theta values.

        Uses numpy searchsorted for O(n log N) binary search across the whole
        array at once, replacing the sequential bisection of _interpolate.

        Args:
            xx_mv: 1-D array of angular distances in radians (far-field only;
                   caller must ensure all values >= 1e-6 and within table range).

        Returns:
            numpy array of interpolated Green's function values, same shape.
        """
        xx   = np.asarray(xx_mv)
        ln_x = np.asarray(self.ln_x)
        y    = np.asarray(self.y)
        y2   = np.asarray(self.y2)

        ln_xx = np.log(xx)

        il = np.searchsorted(ln_x, ln_xx, side='right') - 1
        il = np.clip(il, 0, self.n - 2)
        iu = il + 1

        h  = ln_x[iu] - ln_x[il]
        a  = (ln_x[iu] - ln_xx) / h
        b  = (ln_xx  - ln_x[il]) / h
        yy = (a*y[il] + b*y[iu] +
              ((a**3 - a)*y2[il] + (b**3 - b)*y2[iu]) * (h*h) / 6.0)

        return yy / xx   # undo the y[i] *= x[i] scaling applied at load time
