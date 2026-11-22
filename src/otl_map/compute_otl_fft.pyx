# -*- coding: utf-8 -*-
#
# This computes OTL maps using the Scherneck's FFT approach. Instead of
# computing each cell, each latitude row of the mesh is computed in a
# single process.
#
# It is assumed that the computation and tide grid are the same. In this
# case, the loading contribution of a latitude tide row to a mesh latitude
# row can be written as the convolution of the latitude tide row with a 
# filter which is the Green's function, evaluated at the distance between 
# each tide cell and a single cell on the computation row. Due to symmetry,
# the filter is shift-invariant. Now the application of the filter can be 
# computed using FFT.
# 
#
# This script is part of otl_map 0.0.7
#
# 17/ 3/2023 Coimbra, Santa Clara
#  8/10/2023 Coimbra, Santa Clara
#
# (c) Copyright 2023 TeroMovigo, all rights reserved.
#===============================================================================

import math
import numpy as np
import os
import sys
import json
import netCDF4 as nc
import argparse
from otl_map.greensfunction import Greensfunction
from otl_map.geometry cimport compute_distance
from otl_map.cell_integration cimport integrate_cell,numerical_integral
from otl_map.tidemodel cimport get_tidemodel_mesh, read_tidemodel
from tqdm import tqdm
import contextlib
import datetime
import io
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from libc.math cimport sin,cos,acos,sqrt,atan2,fabs,pi,asin,log
cimport cython # for decorators


#===============================================================================
# Worker function for parallel mesh-row processing
#===============================================================================

def _process_row_batch(args):
    """Process one mesh row in a worker process.

    FU/FV are loaded via memory-mapped .npy files (read-only, OS-paged).
    Greensfunction is reconstructed from the config file — fast, ~100 ms.
    All Cython helpers (compute_distance, integrate_cell, numerical_integral)
    are available because this function is compiled into the same .so.
    """
    cdef int    row_i, j, kk, cnx_, cnharm_, cny_
    cdef double theta, alpha
    cdef double *ptheta = &theta
    cdef double *palpha = &alpha
    cdef double[:] cc_mv
    cdef int    component_
    cdef double lat_m, lat_c, csm, snm, csc, snc
    cdef double A, yy0, yy1, lon_m, lon_c, xp, yp, dx_, dy_

    (row_i, fu_path, fv_path, fname,
     my0, msp1, cny_, cy1, csp0, csp1, cnx_, cnharm_,
     cc_list, rho, scale, hpi, cx0, rad, near_cells) = args

    FU = np.load(fu_path, mmap_mode='r')
    FV = np.load(fv_path, mmap_mode='r')

    # Reconstruct worker-local Greensfunction, suppressing its startup prints.
    with contextlib.redirect_stdout(io.StringIO()):
        gf = Greensfunction(fname)
    component_ = gf.component

    cc_np = np.array(cc_list, dtype=np.double)
    cc_mv = cc_np

    _dlon_arr = np.arange(cnx_ // 2 + 1, dtype=np.double) * csp0
    _cos_dlon = np.cos(_dlon_arr)
    _sin_dlon = np.sin(_dlon_arr)

    lat_m = my0 + row_i * msp1
    csm   = cos(lat_m)
    snm   = sin(lat_m)

    G_np      = np.zeros(cnx_, dtype=np.double)
    contrib_u = np.zeros((cnharm_, cnx_ // 2 + 1), dtype=np.cdouble)
    contrib_v = np.zeros((cnharm_, cnx_ // 2 + 1), dtype=np.cdouble)

    for j in range(cny_):
        lat_c = cy1 - (j + 0.5) * csp1
        csc   = cos(lat_c)
        snc   = sin(lat_c)
        yy0 = lat_c - 0.5 * csp1
        yy1 = lat_c + 0.5 * csp1
        if yy0 < -hpi:
            yy0 = -hpi
        elif yy1 > hpi:
            yy1 = hpi
        A = scale * (sin(yy1) - sin(yy0))

        lon_m  = cx0 + 0.5 * csp0
        csp_k  = snm * snc + csm * csc * _cos_dlon
        np.clip(csp_k, -1.0, 1.0, out=csp_k)
        theta_k = np.arccos(csp_k)

        near_idx = np.where(theta_k < near_cells * csp0)[0]
        far_idx  = np.where(theta_k >= near_cells * csp0)[0]

        if far_idx.size > 0:
            G_far = rho * A * gf.interpolate_array(theta_k[far_idx])
            if component_ != 0:
                snp_k    = np.sqrt(np.maximum(1.0 - csp_k[far_idx]**2, 0.0))
                safe_snp = np.where(snp_k > 0.0, snp_k, 1.0)
                if fabs(csm) < 1e-7:
                    alpha_k = (_dlon_arr[far_idx] - pi) if snm > 0.0 else -_dlon_arr[far_idx]
                else:
                    sna_k   = csc * (-_sin_dlon[far_idx]) / safe_snp
                    csa_k   = (snc - csp_k[far_idx] * snm) / (safe_snp * csm)
                    alpha_k = np.arctan2(sna_k, csa_k)
                if component_ == 1:
                    G_far *= np.cos(alpha_k - pi)
                else:
                    G_far *= np.sin(alpha_k - pi)
            G_np[far_idx] = G_far

        for kk in near_idx:
            lon_c = cx0 + (kk + 0.5) * csp0
            compute_distance(lon_m, csm, snm, lon_c, csc, snc, ptheta, palpha)
            xp  = 6.371e6 * csm * (lon_m - lon_c)
            yp  = 6.371e6 * (lat_m - lat_c)
            dx_ = 6.371e6 * csm * csp0
            dy_ = 6.371e6 * csp1
            if theta + csp0 < 0.15 * rad:
                G_np[kk] = rho * integrate_cell(component_, cc_mv, xp, yp, dx_, dy_)
            else:
                G_np[kk] = rho * numerical_integral(component_, gf, xp, yp, dx_, dy_)

        if component_ == 2:
            G_np[cnx_ // 2 + 1:] = -G_np[1:cnx_ // 2][::-1]
        else:
            G_np[cnx_ // 2 + 1:] =  G_np[1:cnx_ // 2][::-1]

        FG = np.fft.rfft(G_np)
        contrib_u += FG * FU[:, j, :]
        contrib_v += FG * FV[:, j, :]

    return (row_i,
            np.fft.irfft(contrib_u, axis=-1),
            np.fft.irfft(contrib_v, axis=-1))


#===============================================================================
# Main program
#===============================================================================

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def main():

    #--- Start clock
    start_time = datetime.datetime.now()

    #--- Constants 
    cdef double rad = math.pi/180.0
    cdef double R = 6.371e6
    cdef double EPS = 1.0e-4   # absorbs float64 rounding for non-exact spacings like 1/6

    #--- Variables
    cdef int           i,j,k,l,mnx,mny,cnc,cny,cnharm
    cdef double        mx0,mx1,my0,my1,msp0,msp1,rho,csp0,csp1,mxx0
    cdef double        A,lon_c,lat_c,u,v,lat_m,lon_m,csm,snm,csc,snc
    cdef double        theta,alpha,yy0,yy1,scale,hpi,cx0,cy1
    cdef double[:]     cc
    cdef double        *ptheta=&theta
    cdef double        *palpha=&alpha
    cdef double[:,:]   tide_u
    cdef double[:,:]   tide_v
    cdef double[:]     G

    #--- Parse command line arguments in a bit more professional way
    parser = argparse.ArgumentParser(description="compute OTL map with FFT")

    #--- List arguments that can be given
    parser.add_argument('-i', dest='fname', action='store',
           default='otl_map.json', required=False,
			help="The name of the json file with configuration")
    parser.add_argument('--progress', dest='progress', action='store',
           default=None, required=False,
           help="Path to JSON file for progress snapshots (web app mode; disables tqdm)")

    args = parser.parse_args()

    #--- Store the name of the input and output file
    fname         = args.fname
    progress_path = args.progress

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
        [mx0,mx1,my0,my1,msp0,msp1] = configuration['mesh']
    except:
        print('Could not find tidemodel mesh or problem with it')
        sys.exit()
    try:
        harms = configuration['harmonics']
    except:
        harms = ['m2']
    cnharm = len(harms)
    try:
        rho = float(configuration['rho'])
    except:
        rho = 1030.0
    #--- Cells within near_cells grid spacings (angular distance) are integrated over
    #    the cell (analytically inside 0.15 deg, numerically beyond); farther cells
    #    are point loads.  Default 3 = production value.
    near_cells = float(configuration.get('near_cells', 3.0))
    if near_cells != 3.0:
        print('near_cells : {0:f}  (non-default)'.format(near_cells))
    try:
        verbose = bool(configuration['verbose'])
    except:
        verbose = True 
    try:
        label = configuration['label']
    except:
        label = 'fft'
    try:
        ccc = configuration['constants']
        for i in range(0,4):
            cc[i] = ccc[i]
    except:
        print('Cannot find constants approximating greensfnc')
        sys.exit()


    print('OTL map computation (FFT)')
    print('--------------------------')
    print('mesh   : {0:f}, {1:f}, {2:f}, {3:f}, {4:f}, {5:f}'. \
					format(mx0,mx1,my0,my1,msp0,msp1))
    print('rho    : {0:f}'.format(rho))
    print('label  : {0:s}'.format(label))
    print('c[0]   : {0:e}'.format(cc[0]))
    print('c[1]   : {0:e}'.format(cc[1]))
    print('c[2]   : {0:e}'.format(cc[2]))
    print('c[3]   : {0:e}\n'.format(cc[3]))

    #--- Create Green's function class instance
    greensfnc = Greensfunction(fname)

    #--- Get Tide Model mesh info
    [cnx,cny,cx0,cy1,csp0,csp1] = get_tidemodel_mesh(tidemodel,'m2',verbose) 

    #--- Sanity check
    if fabs(cnx*csp0 - 2.0*pi)>EPS:
        print('Only models that cover 360 degrees longitude are acceptable')
        sys.exit()

    #--- Allocate some memory to various arrays
    tide_u  = np.zeros((cnharm,cnx*cny), dtype=np.double)
    tide_v  = np.zeros((cnharm,cnx*cny), dtype=np.double)
    G       = np.zeros(cnx, dtype=np.double)

    #--- Fill tide_u and tide_v
    for l in range(0,cnharm):
        harm = harms[l]
        read_tidemodel(tidemodel,harm,tide_u[l,:],tide_v[l,:],0.0,verbose)

    #--- Pre-compute rfft of every tide row for every harmonic (complex64).
    #    One harmonic at a time: each rfft produces a temporary complex128
    #    array (cny*(cnx//2+1)*16 bytes) before the csingle cast.  Computing
    #    all cnharm at once would make that transient cnharm× larger.
    #    After FU/FV are built the tide_u/tide_v buffers are released.
    #    Shape: (cnharm, cny, cnx//2+1)
    _fu_mb   = 2 * cnharm * cny * (cnx // 2 + 1) * 8 / 1e6
    _tide_mb = 2 * cnharm * cnx * cny * 8 / 1e6
    print('Pre-computing tide FFTs  ({:.0f} MB for FU + FV; '
          'will free {:.0f} MB tide buffers after)…'.format(_fu_mb, _tide_mb))
    FU = np.empty((cnharm, cny, cnx // 2 + 1), dtype=np.csingle)
    FV = np.empty((cnharm, cny, cnx // 2 + 1), dtype=np.csingle)
    for _l in range(cnharm):
        FU[_l] = np.fft.rfft(np.asarray(tide_u[_l]).reshape(cny, cnx), axis=1).astype(np.csingle)
        FV[_l] = np.fft.rfft(np.asarray(tide_v[_l]).reshape(cny, cnx), axis=1).astype(np.csingle)
    # Release the tide buffers — no longer needed; reassign typed memoryviews
    # to a 1×1 array so the large backing arrays can be GC'd.
    tide_u = np.empty((1, 1), dtype=np.double)
    tide_v = np.empty((1, 1), dtype=np.double)
    print('Pre-computation done.\n')

    #--- Compute number of rows and columns
    mnx = int((mx1-mx0)/msp0 + EPS) + 1   
    mny = int((my1-my0)/msp1 + EPS) + 1   

    #--- Write result to netCDF file
    #    int16 + zlib: scale_factor=0.01 gives 0.01 mm / 0.01° resolution;
    #    range ±327 mm / ±327° covers all valid OTL values with headroom.
    #    Write physical float values; netCDF4 auto-packs to int16 via scale_factor.
    nc_amp = [None]*cnharm
    nc_pha = [None]*cnharm
    va     = [None]*cnharm
    vp     = [None]*cnharm
    _chunk = (min(64, mny), min(64, mnx))
    for i in range(0,cnharm):
        harm = harms[i]
        nc_amp[i] = nc.Dataset('amp_fft{0:d}_{1:s}.{2:s}'.format(greensfnc.component,label,harm),'w')
        dx = nc_amp[i].createDimension('x',mnx)
        dy = nc_amp[i].createDimension('y',mny)
        vx = nc_amp[i].createVariable('x', np.double, 'x')
        vy = nc_amp[i].createVariable('y', np.double, 'y')
        vx[:] = np.linspace(mx0,mx1,mnx)
        vy[:] = np.linspace(my0,my1,mny)
        va[i] = nc_amp[i].createVariable('z', 'i2', ('y','x'),
                    zlib=True, complevel=4, chunksizes=_chunk,
                    fill_value=np.int16(-32768))
        va[i].scale_factor = np.float32(0.01)
        va[i].add_offset   = np.float32(0.0)
        va[i].units        = 'mm'
        nc_pha[i] = nc.Dataset('pha_fft{0:d}_{1:s}.{2:s}'.format(greensfnc.component,label,harm),'w')
        dx = nc_pha[i].createDimension('x',mnx)
        dy = nc_pha[i].createDimension('y',mny)
        vx = nc_pha[i].createVariable('x', np.double, 'x')
        vy = nc_pha[i].createVariable('y', np.double, 'y')
        vx[:] = np.linspace(mx0,mx1,mnx)
        vy[:] = np.linspace(my0,my1,mny)
        vp[i] = nc_pha[i].createVariable('z', 'i2', ('y','x'),
                    zlib=True, complevel=4, chunksizes=_chunk,
                    fill_value=np.int16(-32768))
        vp[i].scale_factor = np.float32(0.01)
        vp[i].add_offset   = np.float32(0.0)
        vp[i].units        = 'degrees'

    #--- To avoid always calculating the following numbers, put it in 'scale'
    msp0 *= rad
    msp1 *= rad
    my0  *= rad
    mx0  *= rad
    my1  *= rad
    mx1  *= rad
    hpi   = rad*90.0
    scale = pow(R,2.0)*csp0

    #--- Another sanity check (mesh and ocean grid are aligned?)
    if mx0<cx0+0.5*csp0-EPS:
        mxx0 =  mx0 + 2.0*pi
    else:
        mxx0 =  mx0
    j0 = int((mxx0-cx0-0.5*csp0)/csp0+EPS)
    if fabs((mxx0-cx0-0.5*csp0)/csp0 - j0)>EPS:
        print('j0=',j0,', fraction=',(mxx0-cx0-0.5*csp0)/csp0 - j0)
        print('Oops, mesh grid not aligned with tide grid!')
        sys.exit()

    # Column-offset index: maps output column k → tide-grid column (k+j0) % cnx.
    # Pre-computed here so both the parallel and sequential paths can use it
    # for inline netCDF writes without storing the full otl_u/otl_v buffers.
    jj_idx = (np.arange(mnx) + j0) % cnx

    #--- Pre-compute longitude-offset arrays (constant for every (i,j) iteration).
    #    dlon_arr[k] = k * csp0  = lon_m - lon_c_k  (negated; see compute_distance convention)
    #    cos/sin are hoisted here so the inner loop never recomputes them.
    _dlon_arr = np.arange(cnx // 2 + 1, dtype=np.double) * csp0
    _cos_dlon = np.cos(_dlon_arr)
    _sin_dlon = np.sin(_dlon_arr)   # sin(k*csp0); for alpha: sin(lon_m-lon_c)= -_sin_dlon

    #--- Loop over the list of grid cells for which OTL needs to be computed
    n_workers = max(1, min(mny, int(os.environ.get('OTL_MAP_WORKERS', 0)) or os.cpu_count() or 1))
    if verbose:
        if n_workers > 1:
            print('Parallel: {:d} workers for {:d} mesh rows.'.format(n_workers, mny))
        else:
            print('Sequential mode.')

    if n_workers > 1:
        tmp_dir = tempfile.mkdtemp(prefix='otl_par_')
        try:
            fu_path = os.path.join(tmp_dir, 'FU.npy')
            fv_path = os.path.join(tmp_dir, 'FV.npy')
            np.save(fu_path, np.asarray(FU))
            np.save(fv_path, np.asarray(FV))
            _common = (fu_path, fv_path, fname,
                       float(my0), float(msp1), int(cny), float(cy1),
                       float(csp0), float(csp1), int(cnx), int(cnharm),
                       list(np.asarray(cc)), float(rho), float(scale),
                       float(hpi), float(cx0), float(rad), float(near_cells))
            _all_args = [(i,) + _common for i in range(mny)]
            n_done = 0
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                _futures = {executor.submit(_process_row_batch, a): a[0]
                            for a in _all_args}
                _iter = (tqdm(as_completed(_futures), total=mny)
                         if not progress_path else as_completed(_futures))
                for _fut in _iter:
                    row_i, u_row, v_row = _fut.result()
                    _uc = u_row[:, jj_idx]   # (cnharm, mnx)
                    _vc = v_row[:, jj_idx]
                    _amp = 1000.0 * np.sqrt(_uc**2 + _vc**2)   # mm
                    _pha = np.arctan2(_vc, _uc) / rad            # degrees
                    for _l in range(cnharm):
                        va[_l][row_i, :] = _amp[_l]
                        vp[_l][row_i, :] = _pha[_l]
                    n_done += 1
                    if progress_path:
                        _elapsed = (datetime.datetime.now() - start_time).total_seconds()
                        _eta = (_elapsed / n_done * (mny - n_done)) if n_done > 0 else None
                        _snap = {
                            "pct":       n_done * 100.0 / mny,
                            "done":      n_done,
                            "total":     mny,
                            "elapsed_s": round(_elapsed, 1),
                            "eta_s":     round(_eta, 0) if _eta is not None else None,
                            "msg":       "Row {:d}/{:d}  ({:d} workers)".format(
                                             n_done, mny, n_workers),
                            "phase":     "computing",
                        }
                        with open(progress_path, 'w', encoding='utf-8') as _pf:
                            json.dump(_snap, _pf)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
    else:
        _rows = tqdm(range(mny)) if not progress_path else range(mny)
        for i in _rows:
            if progress_path:
                _elapsed_now = (datetime.datetime.now() - start_time).total_seconds()
                _eta = (_elapsed_now / i * (mny - i)) if i > 0 else None
                _snap = {
                    "pct": i * 100.0 / mny,
                    "done": i,
                    "total": mny,
                    "elapsed_s": round(_elapsed_now, 1),
                    "eta_s": round(_eta, 0) if _eta is not None else None,
                    "msg": "Row {:d} / {:d}".format(i, mny),
                    "phase": "computing",
                }
                with open(progress_path, 'w', encoding='utf-8') as _pf:
                    json.dump(_snap, _pf)
            lat_m = my0 + i*msp1
            csm   = cos(lat_m) # compute_distance only uses cos and sin of lat
            snm   = sin(lat_m)
            contrib_u = np.zeros((cnharm, cnx//2+1), dtype=np.cdouble)
            contrib_v = np.zeros((cnharm, cnx//2+1), dtype=np.cdouble)
            for j in range(cny):

                #--- Area of cell
                lat_c = cy1 - (j+0.5)*csp1
                csc   = cos(lat_c) # compute_distance only uses cos and sin of lat
                snc   = sin(lat_c)
                yy0 = lat_c - 0.5*csp1
                yy1 = lat_c + 0.5*csp1
                if yy0<-hpi:
                    yy0 = -hpi
                elif yy1>hpi:
                    yy1 = hpi
                A = scale*(sin(yy1)-sin(yy0))

                #--- Vectorised Green's function filter ----------------------------
                #
                # For fixed (lat_m, lat_c), cos(theta_k) = snm*snc + csm*csc*cos(k*csp0)
                # for k=0..cnx//2.  Compute all at once with numpy, then split into
                # far-field (vectorised spline) and near-field (element-by-element).
                #
                lon_m = cx0+0.5*csp0

                csp_k = snm*snc + csm*csc*_cos_dlon      # cos(theta) for each k
                np.clip(csp_k, -1.0, 1.0, out=csp_k)
                theta_k = np.arccos(csp_k)               # theta[k], radians

                near_idx = np.where(theta_k < near_cells*csp0)[0]
                far_idx  = np.where(theta_k >= near_cells*csp0)[0]
                G_np     = np.asarray(G)

                # Far-field: vectorised spline, optional azimuth for horiz. components
                if far_idx.size > 0:
                    G_far = rho * A * greensfnc.interpolate_array(theta_k[far_idx])
                    if greensfnc.component != 0:
                        snp_k   = np.sqrt(np.maximum(1.0 - csp_k[far_idx]**2, 0.0))
                        safe_snp = np.where(snp_k > 0.0, snp_k, 1.0)
                        if fabs(csm) < 1e-7:        # at a pole
                            if snm > 0.0:
                                alpha_k = _dlon_arr[far_idx] - pi
                            else:
                                alpha_k = -_dlon_arr[far_idx]
                        else:
                            # sin(lon_m - lon_c_k) = sin(-k*csp0) = -_sin_dlon[k]
                            sna_k   = csc * (-_sin_dlon[far_idx]) / safe_snp
                            csa_k   = (snc - csp_k[far_idx]*snm) / (safe_snp * csm)
                            alpha_k = np.arctan2(sna_k, csa_k)
                        if greensfnc.component == 1:
                            G_far *= np.cos(alpha_k - pi)
                        else:
                            G_far *= np.sin(alpha_k - pi)  # half-filter: two sign flips
                    G_np[far_idx] = G_far

                # Near-field: element-by-element (typically 0–5 cells per (i,j))
                for kk in near_idx:
                    lon_c = cx0 + (kk+0.5)*csp0
                    compute_distance(lon_m,csm,snm,lon_c,csc,snc,ptheta,palpha)
                    xp = R*csm*(lon_m-lon_c)
                    yp = R*(lat_m-lat_c)
                    dx = R*csm*csp0
                    dy = R*csp1
                    if theta + csp0 < 0.15*rad:
                        G_np[kk] = rho*integrate_cell(greensfnc.component,cc,xp,yp,dx,dy)
                    else:
                        G_np[kk] = rho*numerical_integral(greensfnc.component,greensfnc,xp,yp,dx,dy)

                # Symmetry: copy G[1..cnx//2-1] → G[cnx-1..cnx//2+1] (reversed)
                if greensfnc.component == 2:   # East-West: anti-symmetric
                    G_np[cnx//2+1:] = -G_np[1:cnx//2][::-1]
                else:
                    G_np[cnx//2+1:] =  G_np[1:cnx//2][::-1]

                #--- FFT of Green's function filter; convolve with pre-computed
                #    tide FFTs (complex64) for all harmonics at once.
                #    numpy up-casts FU[l,j] to complex128 before multiplying with
                #    FG, so full double-precision is preserved in the convolution.
                FG = np.fft.rfft(G)
                contrib_u += FG * FU[:, j, :]
                contrib_v += FG * FV[:, j, :]

            # irfft(sum_j FG_j * FU_j) = sum_j irfft(FG_j * FU_j): one irfft per
            # harmonic per mesh row instead of cny×cnharm irffts.
            _u_row = np.fft.irfft(contrib_u, axis=-1)   # (cnharm, cnx)
            _v_row = np.fft.irfft(contrib_v, axis=-1)
            _uc = _u_row[:, jj_idx]                      # crop to requested lon range
            _vc = _v_row[:, jj_idx]
            _amp = 1000.0 * np.sqrt(_uc**2 + _vc**2)    # mm
            _pha = np.arctan2(_vc, _uc) / rad             # degrees
            for _l in range(cnharm):
                va[_l][i, :] = _amp[_l]
                vp[_l][i, :] = _pha[_l]


    # Results were written row-by-row during the main loop; just flush and close.
    print('\nFlushing NetCDF output files...')
    for l in range(cnharm):
        nc_amp[l].close()
        nc_pha[l].close()


    #--- Stop the clock
    end_time = datetime.datetime.now()
    print('\nTotal computation time : {0:}'.format(end_time - start_time))

    if progress_path:
        _end_elapsed = (end_time - start_time).total_seconds()
        _snap = {
            "pct": 100.0,
            "done": mny,
            "total": mny,
            "elapsed_s": round(_end_elapsed, 1),
            "eta_s": 0,
            "msg": "Done",
            "phase": "done",
        }
        with open(progress_path, 'w', encoding='utf-8') as _pf:
            json.dump(_snap, _pf)
