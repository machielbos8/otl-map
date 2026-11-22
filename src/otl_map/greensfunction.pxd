# -*- coding: utf-8 -*-
#
# This file contains the class "Greensfunction" for reading the Green's
# function into memory and provide an subroutine for getting its value
# at any given angular distance.
#
# This script is part of otl_map 0.0.1
#
#  9/3/2023 Coimbra, Santa Clara
#  6/8/2023 Coimbra, Santa Clara
#
# (c) Copyright 2023 TeroMovigo, all rights reserved.
#===============================================================================



#===============================================================================
# Class definition
#===============================================================================


cdef class Greensfunction:
    ''' Class to read Green's function and provide interpolation subroutine
    '''

    cdef public int    component
    cdef unsigned int il,iu,n
    cdef public double rad
    cdef public bint verbose
    cdef public t_spline
    cdef public double[:] x
    cdef public double[:] y
    cdef public double[:] y2
    cdef public double[:] ln_x

    cdef initialise_spline(self,double[:] x, double[:] y)
 
    cpdef double spline(self, unsigned int il, double xx)

    cpdef double _interpolate(self,double xx)

    cpdef double interpolate(self, double xx)
