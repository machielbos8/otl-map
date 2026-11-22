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


#===============================================================================
# Subroutine definitions
#===============================================================================


cdef get_tidemodel_mesh(tidemodel,harm,verbose)

cdef read_tidemodel(tidemodel,harm,u,v,land_value,verbose)

cdef write_tidemodel(tidemodel,harm,u,v,xr,yr,sp,nx,ny)
