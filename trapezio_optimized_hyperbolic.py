#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on  June 2023

@author: M. Calvo, A. Elipe & L. Rández

trpz_opt :  Use the trapezoidal composite rule for computing the integrals 
            in the numerator and denominator in [0,pi] for solving the 
            hyperbolic Kepler equation  
                  excen * sinh(z) - z = time_array. 
                 
"""
import numpy as np

def trpz_opt(time_array, N_fft=32, excen=1.1, epsi=1):
    """    
    This function is optimized in the interval [0, pi] instead of [0,2pi]
    
    Args:
        time_array (np.ndarray): Array of times hyperbolic anomaly.
        
    Keyword Args:
        N_fft (int)   : Number of FFT points, default: 32.
        excen (float) : value of hyperbolic eccentricity
        epsi (float)  : Small parameter used to set the mayor semiaxis of the 
                        ellipse relative to the Jordan contour, default 1 (circular).

    Returns:
        np.ndarray: solution of the transcendental equation 
                 excen * sinh(z) - z = time_array. 
    """
# Input checks
    assert N_fft > 1, "Need at least two FFT grid-point!"
    assert excen > 1, "eccentricity must be grater than one!"
#    
# Define contour integrand        
#
    def gnueva(x, t, excen):
        return 1./(excen*np.sinh(x) - x - t)
#
# Create FFT array and evaluate g(x;t)
# bounds for the coordinates (e,M) 
#
    a  = time_array
    c1 = a[a<=np.sqrt(6)*(excen-1)**1.5/np.sqrt(np.e)]; 
    a  = a[a>np.sqrt(6)*(excen-1)**1.5/np.sqrt(np.e)]
    c2 = a[a<=14.907*excen]; a=a[a>14.907*excen]
    c3 = a
    c1 = c1/(excen-1)
    c2 = (6*c2/excen)**0.3333333333333
    c3 = (120*c3/excen)**0.2
#
# Add more bounds ci if  M > 95.2669*excen. See the paper
#    
    xmas   = np.hstack((c1,c2,c3))
    xmenos = np.arcsinh( time_array/excen )
    mu  = (xmas + xmenos)/2
    rho = (xmas - xmenos)/2
    x   = np.linspace(0, np.pi, N_fft)[:,np.newaxis]
    si  = np.sin(x); co = np.cos(x)
    aux = mu + rho*(co + 1j*epsi*si)
    G   = gnueva(aux, time_array[np.newaxis,:], excen)
    tablaD = (1j*epsi*co - si)*G*rho
    tablaN = aux*tablaD

    numer = tablaN[0] + tablaN[-1] + 2*np.sum( tablaN[1:N_fft-1],axis=0)
    denom = tablaD[0] + tablaD[-1] + 2*np.sum( tablaD[1:N_fft-1],axis=0)

    z0_array = np.imag(numer)/np.imag(denom)
    return z0_array

#
#
#
npuntos = 10
N_fft = 32
sol = trpz_opt(np.linspace(0.1, 20, npuntos), N_fft, excen=2, epsi=1)
print( sol )
