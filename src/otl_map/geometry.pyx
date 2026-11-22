# -*- coding: utf-8 -*-
#
# This file contains two subroutines which computes angular distances
# between to two points on a sphere and its inverse operation.  It is
# written in Cython.
#
# Point 1 is where we define angle alpha, the azimuth. Assume this is the
# stations that feels the effect of a water cell at point 2. 
#
# This script is part of otl_map 0.0.7
#
# 22/3/2023 Coimbra, Santa Clara
#  6/8/2023 Coimbra, Santa Clara
#
# (c) Copyright 2023 TeroMovigo, all rights reserved.
#===============================================================================

import math
from libc.math cimport sin,cos,acos,sqrt,atan2,fabs,pi,asin

#===============================================================================
# Subroutines
#===============================================================================

cdef void compute_distance(double lon1, double cs1, double sn1, double lon2,\
		        double cs2, double sn2, double* beta, double* alpha):
    cdef:
        double csp,snp,sna,csa

    csp = sn1*sn2 + cs1*cs2*cos(lon1-lon2)
    snp = sqrt(1.0-csp*csp)
    if csp>0.999999999:
        beta[0] = alpha[0] = 0.0  # distance to station is zero
    elif csp<-0.999999999:
        beta[0]  = pi
        alpha[0] = 0.0
    elif fabs(cs1)<1.0e-7:
        beta[0]  = acos(csp)
        if sn1>0.0:
            alpha[0] = lon2-lon1-pi
        else:
            alpha[0] = lon1-lon2
    else:
        beta[0] = acos(csp)
        sna = cs2*sin(lon1-lon2)/snp
        csa = (sn2-csp*sn1)/(snp*cs1)
        alpha[0] = atan2(sna,csa)



cdef void compute_lonlat(double lon1, double lat1, \
		double beta, double alpha, double* lon2, double* lat2):
    cdef:
        double sn1,cs1,sn,cs,csp,snp,snm,csm

    sn  = sin(beta)
    cs  = cos(beta)
    cs1 = cos(lat1)
    sn1 = sin(lat1)
    snp = cs*sn1+sn*cs1*cos(alpha)
    lat2[0] = asin(snp)
    if fabs(cs1)<1.0e-7:
        if lat1>0.0:
            lon2[0] = alpha+lon1+pi
        else:
            lon2[0] = lon1-alpha
    else:
        #--- I have removed csp in the fractions of snm and csm to avoid
        #    the singularity for csp==0.0. In atan2 it cancels out anyway.
        snm = sin(alpha)*sn
        csm = (cs - snp*sn1)/cs1
        lon2[0] = lon1 - atan2(snm,csm)
