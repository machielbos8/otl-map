# -*- coding: utf-8 -*-
#
# This file contains DEFINITIONS of subroutines which computes angular 
# distances between to two points on a sphere and its inverse operation. 
# It is written in Cython.
#
# Point 1 is where we define angle alpha, the azimuth. Assume this is the
# stations that feels the effect of a water cell at point 2. 
#
# This script is part of otl_map 0.0.7
#
#  6/8/2023 Coimbra, Santa Clara
#
# (c) Copyright 2023 TeroMovigo, all rights reserved.
#===============================================================================

cdef void compute_distance(double lon1, double cs1, double sn1, double lon2,\
                        double cs2, double sn2, double* beta, double* alpha)

cdef void compute_lonlat(double lon1, double lat1, \
                   double theta, double alpha, double* lon2, double* lat2)
