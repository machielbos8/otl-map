# -*- coding: utf-8 -*-
#
# This file contains DEFINITIONS of subroutines 
# distances between to two points on a sphere and its inverse operation. 
# It is written in Cython.
#
# This script is part of otl_map 0.0.7
#
# 11/9/2023 Coimbra, Santa Clara
#
# (c) Copyright 2023 TeroMovigo, all rights reserved.
#===============================================================================

import math
import numpy as np
import os
import sys
from libc.math cimport sin,cos,acos,sqrt,atan2,fabs,pi,asin,log
from otl_map.greensfunction cimport Greensfunction

#===============================================================================
# Subroutines
#===============================================================================

cdef double indefinite_integral_ver(double[:] c, double x, double y)

cdef double integrate_ver(double[:] c, double xp, double yp, \
                                                        double dx, double dy)

cdef double integrate_hor(double[:] c, double xp, double yp, \
                                                        double dx, double dy)

cpdef double integrate_cell(int component, double[:] c, double xp, double yp, \
							double dx, double dy)

cdef double numerical_integral_ver(Greensfunction gf, double xp, double yp,\
                                                        double dx, double dy)

cdef double numerical_integral_hor(Greensfunction gf, double xp, double yp,\
                                                        double dx, double dy)

cpdef double numerical_integral(int component, Greensfunction gf, double xp,\
					     double yp, double dx, double dy)
                                                       
