# -*- coding: utf-8 -*-
#
# This script fits a pole and second degree polynomial to the vertical
# or horizontal component of a Green's function (point load). 
#
# This script is part of otl_map 0.0.7
#
# 20/9/2023 Coimbra, Santa Clara
#
# (c) Copyright 2023 TeroMovigo, all rights reserved.
#===============================================================================

import os
from otl_map.paths import greenfnc_dir, tidemodel_dir
import sys
import json
import numpy as np
from matplotlib import pyplot as plt
import argparse
import math
from numpy.linalg import inv


#===============================================================================
# Main program
#===============================================================================

def main():

    #--- Parse command line arguments in a bit more professional way
    parser = argparse.ArgumentParser(description="fit pole + polynomial")

    #--- List arguments that can be given 
    parser.add_argument('-g', dest='greensfnc', action='store', required=False,\
          	default='STW105', help="Green's function name") 
    parser.add_argument('-c', dest='component', action='store',\
		default='0', required=True, help='Component 0=ver, 1=hor')
    parser.add_argument('-o', dest='fname_out', action='store',\
          	default='fit_pointload.json', required=False, \
		help='name of output json file')

    args = parser.parse_args()

    #--- Store the name of the input and output file
    greensfnc = args.greensfnc
    component = int(args.component)
    fname_out = args.fname_out

    greensfnc_fname = greensfnc + '.dat'
    
    #--- Read Green's function
    directory = greenfnc_dir()
    try:
        fp_in = open(directory + '/' + greensfnc_fname,'r')
    except:
        print('Could not open {0:s}'.format(
                                        directory + '/' + greensfnc_fname))
        sys.exit()

    line = fp_in.readline()
    
    print("\nRead Green's function\n---------------------")
    print("file      : {0:s}".format(greensfnc_fname))
    print("header    : {0:s}".format(line.strip()))
    print("component : {0:d}".format(component))
                                        
    #--- constants
    R = 6.371e6
    rad = math.pi/180

    t = []
    y = []
    for line in fp_in:
        cols = line.split()
        theta = float(cols[0])
        if theta>1.0e-5 and theta<0.2:
            t.append(theta)
            y.append(float(cols[1+component]))

    #---- create design matrix
    n = len(t)
    H = np.ones((n,4))
    for i in range(0,n):
        H[i,0] = 1/(R*rad*t[i])
        H[i,2] = pow(R*rad*t[i],1) 
        H[i,3] = pow(R*rad*t[i],2) 

    #--- give more weight to the larger distances   
    Hw = np.zeros((n,4))
    yw = np.zeros(n)
    for i in range(0,n):
        for j in range(0,4):
            Hw[i,j] = H[i,j]*t[i]
        yw[i] = y[i]*t[i]

    #--- Perform OLS (weighting is hidden)
    theta = np.linalg.lstsq(Hw, yw, rcond=None)[0]
    print(theta)
    y_hat = H @ theta

    ys = np.zeros(n)
    ys_hat = np.zeros(n)
    s = np.zeros(n)
    for i in range(0,n):
        s[i] = 0.001 * R*rad * t[i]
        ys[i]     = y[i] * R*rad*1.0e12*t[i]
        ys_hat[i] = y_hat[i] * R*rad*1.0e12*t[i]

    #--- I now want to try to make all plots using Computer Modern font
    plt.rcParams['text.usetex'] = True
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'

    fig = plt.figure(figsize=(6, 2.5), dpi=150)

    
    i0 = 60
    if component==0:
        plt.semilogx(s[i0:], ys[i0:],     'k-', label=r'{0:s} (radial)'.\
							format(greensfnc))
    else:
        plt.semilogx(s[i0:], ys[i0:],     'k-', label=r'{0:s} (horizontal)'.\
							format(greensfnc))
    plt.semilogx(s[i0:], ys_hat[i0:], 'r-', label=r'fitted function')
        
    plt.legend()
    plt.xlabel('distance (km)')
    plt.ylabel(r'G $\times\; a/m_e \times 10^{12}$R$\phi$  (m/kg)')

    plt.show()

    #--- Save png
    if component==0:
        fig.savefig('fitted_Gver.eps', format='eps', bbox_inches='tight')
    else:
        fig.savefig('fitted_Ghor.eps', format='eps', bbox_inches='tight')

    #--- save results to file
    output = {}
    output['component'] = component
    output['greensfunction'] = greensfnc_fname
    output['constants'] = theta.tolist()

    #--- Serializing json
    json_object = json.dumps(output, indent=4)
 
    #--- Writing to sample.json
    with open(fname_out, "w") as fp:
        fp.write(json_object)
                                       


if __name__=='__main__':
    main()
