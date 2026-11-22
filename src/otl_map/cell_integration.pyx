# -*- coding: utf-8 -*-
#
# This script itegrates the approximated Green's function over a rectangular
# cell of dimenions dx / dy. The station is assumed to be at point xp, yp.
#
# This script is part of otl_map 0.0.7
#
# 11/ 9/2023 Coimbra, Santa Clara
#  8/10/2023 Coimbra, Santa Clara
#
# (c) Copyright 2023 TeroMovigo, all rights reserved.
#===============================================================================

import math
import numpy as np
import os
import sys
from libc.math cimport sin,cos,acos,sqrt,atan2,fabs,pi,asin,log
cimport numpy as cnp
from otl_map.greensfunction cimport Greensfunction

#===============================================================================
# Subroutines
#===============================================================================

cdef double indefinite_integral_ver(double[:] c, double x, double y):
    """ Wolfram's beatiful analytical expression for the integral

    Args:
        x (double) : cartesian distance away from center of cell
        y (double) : cartesian distance away from center of cell

    Returns:
        (float) : indefinite integral of 1/r function 
    """
 
    cdef double r,x2,x3,y2,y3

    x2 = x*x
    x3 = x2*x
    y2 = y*y
    y3 = y2*y

    #--- simply pythagoras
    r = sqrt(x*x + y*y)

    if abs(x)<1.0e-4:
        part0 = 0.0
    elif abs(y)<1.0e-4:
        part0 = x*math.log(y+r)
    else:
        part0 = 0.5*(y*math.log((1+x/r)/(1-x/r)) + 2*x*math.log(y+r))

    part1 = x*y

    if abs(x)<1.0e-4 and abs(y)<1.0e-4:
        part2=0.0
    elif abs(x)<1.0e-4:
        part2=1/12*(2*y3*math.log(r+x) - 2*y3/3)
    elif abs(y)<1.0e-4:
        part2 = 1/12*(3*x3*math.log(r+y))
    else:
        part2=1/12*((4*x*y3+x3*x*math.sqrt(y2/x2+1)*np.arcsinh(y/x)+4*x3*y)/r-\
                2*x3*np.arctanh(y/r)+\
                        2*y3*math.log(r+x) + 3*x3*math.log(r+y) - 2*y3/3)
        
    part3 = x3*y/3.0 + x*y3/3.0


    return (c[0]*part0 + c[1]*part1 + c[2]*part2 + c[3]*part3)



cdef double integrate_ver(double[:] c, double xp, double yp, \
							double dx, double dy):
    """

    Args:
        xp (double) : cartesian distance of station away from center of cell
        yp (double) : cartesian distance of station away from center of cell
        dx (double) : with of cell
        dy (double) : height of cell
   
    Returns:
        (double) : the integral of 1/r over the cell where r is distance 
			between water and station at xp,yp.
    """
  
    cdef double f1,f2

    f1 = indefinite_integral_ver(c,dx/2-xp, dy/2-yp) - \
			indefinite_integral_ver(c,-dx/2-xp, dy/2-yp)
    f2 = indefinite_integral_ver(c,dx/2-xp,-dy/2-yp) - \
			indefinite_integral_ver(c,-dx/2-xp,-dy/2-yp)
                                
    return f1-f2



cdef double indefinite_integral_hor(double[:] c, double x, double y):
    """ Wolfram's beatiful analytical expression for the integral

    Args:
        x (double) : cartesian distance away from center of cell
        y (double) : cartesian distance away from center of cell

    Returns:
        (float) : indefinite integral 
    """
 
    cdef double r,x2,x3,y2,y3

    x2 = x*x
    x3 = x2*x
    y2 = y*y
    y3 = y2*y

    #--- simply pythagoras
    r = sqrt(x2 + y2)
    r2 = r*r

    if abs(x)<1.0e-4 and abs(y)<1.0e-4:
        return 0.0

    if abs(x)<1.0e-4:
        part0 = -0.5*y*(math.log(r2) - 2.0)
    else:
        part0 = -0.5*(y*(math.log(r2) - 2.0) + 2*x*math.atan(y/x))

    if abs(x)<1.0e-4:
        part1 = -0.5*y*r
    else:
        part1 = -0.5*(y*r + x2*math.log(y+r))

    part2 = -0.5*x2*y

    if abs(x)<1.0e-4:
        part3 = -r*y3/12.0         # limit x -> 0 of the expression below (was 0.0, which dropped 2*y3)
    else:
        part3= -r/24*(5*x2*y + 3*x3/math.sqrt(y2/x2+1)*np.arcsinh(y/x)+2*y3)

    return (c[0]*part0 + c[1]*part1 + c[2]*part2 + c[3]*part3)



cdef double integrate_hor(double[:] c, double xp, double yp, \
							double dx, double dy):
    """

    Args:
        xp (double) : cartesian distance of station away from center of cell
        yp (double) : cartesian distance of station away from center of cell
        dx (double) : with of cell
        dy (double) : height of cell
   
    Returns:
        (double) : the integral of 1/r over the cell where r is distance 
			between water and station at xp,yp.
    """
  
    cdef double f1,f2

    f1 = indefinite_integral_hor(c,dx/2-xp, dy/2-yp) - \
			indefinite_integral_hor(c,-dx/2-xp, dy/2-yp)
    f2 = indefinite_integral_hor(c,dx/2-xp,-dy/2-yp) - \
			indefinite_integral_hor(c,-dx/2-xp,-dy/2-yp)
                                
    return f1-f2



cpdef double integrate_cell(int component, double[:] c, double xp, double yp,\
							double dx, double dy):
    """

    Args:
        c (double) : scaling constants
        xp,yp (double)    : position in metres of the station
        dx,dy (double)    : width and height of the cell in m

    Return:
        Ga*integrate_pole + Gb*dx*dy + Gc*integrate_distc
    """

    #--- FFT assumes a delay while here x increases. Thus, we need to invert
    #    the sign of x. Furthermore, if we rotate the axis 90 degrees to 
    #    get the north component, we again need to change the sign of yp which
    #    thus cancels out.

    if component==0:
        return integrate_ver(c,-xp,yp,dx,dy)
    elif component==1:
        return integrate_hor(c,yp,xp,dy,dx)
    else:
        return integrate_hor(c,-xp,yp,dx,dy)



cdef double numerical_integral_ver(Greensfunction gf, double xp, double yp,\
							double dx, double dy):
    """
    """

    cdef double  R = 6.371e6
    cdef double  rad = pi/180

    Gint_check = 0.0
    mm = int(300.0 * dx/(R*rad*0.0625))  
    if mm<50:
        mm = 50
    #print('mm = {0:d}'.format(mm))
    ddx = dx/mm
    ddy = dy/mm
    for ii in range(0,mm):
        xx = -dx/2.0 + (ii+0.5)*ddx - xp
        for jj in range(0,mm):
            yy = -dy/2.0 + (jj+0.5)*ddy - yp
            rr = math.sqrt(xx*xx + yy*yy)
            theta = rr/R
            if theta>1.0e-12:
                Gint_check += gf.interpolate(theta)
            else:
                print('rr=',rr)
                print('theta=',theta)

    return Gint_check*ddx*ddy



cdef double numerical_integral_hor(Greensfunction gf, double xp, double yp,\
							double dx, double dy):
    """
    """

    cdef double  R = 6.371e6
    cdef double  rad = pi/180

    Gint_check = 0.0
    mm = int(300.0 * dx/(R*rad*0.0625))  
    if mm<50:
        mm = 50
    ddx = dx/mm
    ddy = dy/mm
    for ii in range(0,mm):
        xx = -dx/2.0 + (ii+0.5)*ddx - xp
        for jj in range(0,mm):
            yy = -dy/2.0 + (jj+0.5)*ddy - yp
            rr = math.sqrt(xx*xx + yy*yy)
            theta = rr/R
            if theta>1.0e-12:
                Gint_check -= gf.interpolate(theta) * xx/rr
            else:
                print('rr=',rr)
                print('theta=',theta)

    return Gint_check*ddx*ddy






cpdef double numerical_integral(int component, Greensfunction gf, double xp, \
					     double yp, double dx, double dy):
						
    """

    Args:
        gf (Greensfunction) : Greensfunction class instance
        xp,yp (double)      : position in metres of the station
        dx,dy (double)      : width and height of the cell in m

    Return:
        numerical integration over cell
    """

    #--- FFT assumes a delay while here x increases. Thus, we need to invert
    #    the sign of x. Furthermore, if we rotate the axis 90 degrees to 
    #    get the north component, we again need to change the sign of yp which
    #    thus cancels out.

    if component==0:
        return numerical_integral_ver(gf,-xp,yp,dx,dy)
    elif component==1:
        return numerical_integral_hor(gf,yp,xp,dy,dx)
    else:
        return numerical_integral_hor(gf,-xp,yp,dx,dy)

