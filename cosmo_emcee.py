import numpy as np
import matplotlib.pyplot as plt 
import pyccl as ccl
import scipy.stats as stats
import scipy.integrate as integ
import scipy.optimize as opt
import scipy.special as spec
from scipy.misc import derivative
import sympy as sp
import pandas as pd
import numpy as np
import re
import emcee
import os
import time
import timeit
import tqdm
import multiprocessing
from multiprocessing import Pool
multiprocessing.set_start_method("fork")
import h5py

import os

os.environ["OMP_NUM_THREADS"] = "1"


#number of bin and multipoles
bin_l = 20
bin_euclid=13
#muktipoles logarithmic array and difference definition
ell = np.geomspace(2, 2000, bin_l)
delta_ell = np.empty([bin_l])
delta_ell[1:bin_l] = -ell[0:bin_l-1] + ell[1:bin_l]
delta_ell[0]=ell[0]
#redshift e bin data loading
#euclid total
#bin normalization and nz values
norm_tot= np.zeros(bin_euclid)
nz_tot= np.zeros([bin_euclid,1000])
#redshift and noramlization values
z_eff= np.zeros(bin_euclid)
z_euclid = np.loadtxt("/home/systembio/cosmodata/nofzs/nofz_13_bins_EP_24.5_maxmag_0.2_2.5_zmin_zmax.txt", delimiter="\t", usecols=0)
norm_tot= np.loadtxt("/home/systembio/cosmodata/nofzs/nofz_13_bins_EP_24.5_maxmag_0.2_2.5_zmin_zmax_number_densities.txt")
#galaxy bias for nz total
bias_tot= np.loadtxt("/home/systembio/cosmodata/nofzs/nofz_13_bins_EP_24.5_maxmag_0.2_2.5_zmin_zmax_galaxy_bias.txt",usecols=1)
for i in range(bin_euclid):
    nz_tot[i,:]=np.loadtxt("/home/systembio/cosmodata/nofzs/nofz_13_bins_EP_24.5_maxmag_0.2_2.5_zmin_zmax.txt", delimiter="\t", usecols=i+1)
#euclid red
#bin normalization and nz values
norm_red= np.zeros(bin_euclid)
nz_red= np.zeros([bin_euclid,1000])
#redshift and noramlization values
z_euclid = np.loadtxt("/home/systembio/cosmodata/nofzs/nofz_red_13_bins_EP_24.5_maxmag_0.2_2.5_zmin_zmax.txt", delimiter="\t", usecols=0)
norm_red= np.loadtxt("/home/systembio/cosmodata/nofzs/nofz_red_13_bins_EP_24.5_maxmag_0.2_2.5_zmin_zmax_number_densities.txt")
#galaxy bias for nz red
bias_red= np.loadtxt("/home/systembio/cosmodata/nofzs/nofz_red_13_bins_EP_24.5_maxmag_0.2_2.5_zmin_zmax_galaxy_bias.txt",usecols=1)
for i in range(bin_euclid):
    nz_red[i,:]=np.loadtxt("/home/systembio/cosmodata/nofzs/nofz_red_13_bins_EP_24.5_maxmag_0.2_2.5_zmin_zmax.txt", delimiter="\t", usecols=i+1)
#euclid blue
#bin normalization and nz values
norm_blue= np.zeros(bin_euclid)
nz_blue= np.zeros([bin_euclid,1000])
#redshift and noramlization values
z_euclid = np.loadtxt("/home/systembio/cosmodata/nofzs/nofz_blue_13_bins_EP_24.5_maxmag_0.2_2.5_zmin_zmax.txt", delimiter="\t", usecols=0)
norm_blue= np.loadtxt("/home/systembio/cosmodata/nofzs/nofz_blue_13_bins_EP_24.5_maxmag_0.2_2.5_zmin_zmax_number_densities.txt")
#galaxy bias for nz red
bias_blue= np.loadtxt("/home/systembio/cosmodata/nofzs/nofz_blue_13_bins_EP_24.5_maxmag_0.2_2.5_zmin_zmax_galaxy_bias.txt",usecols=1)
for i in range(bin_euclid):
    nz_blue[i,:]=np.loadtxt("/home/systembio/cosmodata/nofzs/nofz_blue_13_bins_EP_24.5_maxmag_0.2_2.5_zmin_zmax.txt", delimiter="\t", usecols=i+1)
    
    
    
    
    
#sky fractions
fsky_eu = 0.36
#variance of the observed ellipticities
se = 0.3
# IAs parameters fiducial values
eta_red = 0.
eta_blue = 0.
z_pivot=0.62
A_red=2.15
A_blue=0.03
C_1=5e-14
#Comsmological paramneter fiducial values
Oc=0.27
s8=0.83
Omega_b=0.045
h = 0.67
n_s = 0.96
w0 = -1.0
wa = 0.0
# Cl angular power spectra calculator
def cl_calc(Oc,s8,Omega_b,h,n_s,w0,wa,A_red,A_blue,eta_red,eta_blue):
    cosmo = ccl.Cosmology(Omega_c=Oc,
                      Omega_b = Omega_b,
                      h=h,
                      n_s=n_s,
                      sigma8=s8,
                      w0 = w0,
                      wa = wa,
                      transfer_function='boltzmann_camb',
                      extra_parameters = {"camb": {"dark_energy_model": "ppf"}})
    D= ccl.growth_factor(cosmo, 1./(1+z_euclid))
    rho_m = ccl.physical_constants.RHO_CRITICAL * (cosmo['Omega_c']+cosmo['Omega_b'])
    Az_red =  A_red* C_1 * (rho_m / D)*((1+z_euclid)/(1+z_pivot))**eta_red
    Az_blue = 5.*A_blue* C_1 * (rho_m / (D**2))*((1+z_euclid)/(1+z_pivot))**eta_blue
    zeros_l = np.zeros(bin_l)
    clbtracer_eu_blue = np.array([ ccl.WeakLensingTracer(cosmo, dndz=(z_euclid, nz_blue[i,:]), ia_bias=(z_euclid, Az_blue), use_A_ia=False) for i in range(bin_euclid)])
    clbtracer_eu_red = np.array([ ccl.WeakLensingTracer(cosmo, dndz=(z_euclid, nz_red[i,:]), ia_bias=(z_euclid, Az_red), use_A_ia=False) for i in range(bin_euclid)])
    clbtracer = np.concatenate((clbtracer_eu_blue,clbtracer_eu_red))

    clij_smart = np.array([ [
        ccl.angular_cl(cosmo, clbtracer[i], clbtracer[j], ell)
        if j<=i else np.zeros(bin_l)
        for j in range(bin_euclid*2) ] for i in range(bin_euclid*2) ]).T
    clij_smart += np.triu(clij_smart, 1).transpose(0, 2, 1)
    return clij_smart

clij_smart= cl_calc(Oc,s8,Omega_b,h,n_s,w0,wa,A_red,A_blue,eta_red,eta_blue)
# noise and variance coefficient calculation
zeros_matrix = np.zeros([bin_l,bin_euclid,bin_euclid])
ones_matrix = np.ones([bin_l,bin_euclid,bin_euclid])
noise_blue= (se**2/((norm_blue*(60.*180./(np.pi))**2.)))
noise_red= (se**2/((norm_red*(60.*180./(np.pi))**2.)))
noise_matrix_blue= np.zeros((bin_euclid,bin_euclid), float)
noise_matrix_red= np.zeros((bin_euclid,bin_euclid), float)
np.fill_diagonal(noise_matrix_blue, noise_blue)
np.fill_diagonal(noise_matrix_red, noise_red)
noise_matrix_blue = np.array([ noise_matrix_blue for l in range(bin_l)])
noise_matrix_red = np.array([ noise_matrix_red for l in range(bin_l)])

noise = np.block([

             [noise_matrix_blue[:],                 zeros_matrix[:]],

             [zeros_matrix[:],                  noise_matrix_red[:]]

             ])
coeff =  np.block([

             [ones_matrix[:]*np.sqrt(2./((2.*ell[:, None, None]+1.)*delta_ell[:, None, None]*fsky_eu)),                 ones_matrix[:]*np.sqrt(2./((2.*ell[:, None, None]+1.)*delta_ell[:, None, None]*fsky_eu))],

             [ones_matrix[:]*np.sqrt(2./((2.*ell[:, None, None]+1.)*delta_ell[:, None, None]*fsky_eu)),                 ones_matrix[:]*np.sqrt(2./((2.*ell[:, None, None]+1.)*delta_ell[:, None, None]*fsky_eu))]

             ])


# Cl flattening and covariance matrix calculation
clij_smart_flat = np.array([[clij_smart[l,i,j] for i in range(bin_euclid*2) for j in range(i,bin_euclid*2) ]for l in range(bin_l)])
aij = np.array([coeff[l,:,:]*(clij_smart[l,:,:]+noise[l,:,:]) for l in range(bin_l)])

inv_aij =  np.array([ np.linalg.inv(aij[l,:,:]) for l in range(bin_l) ])
sigma = np.array([[0.5*aij[:,i,k]*aij[:,j,l] + 0.5*aij[:,i,l]*aij[:,j,k] for i in range(bin_euclid*2) for j in range(i,bin_euclid*2)] for k in range(bin_euclid*2) for l in range(k,bin_euclid*2)])
inv_sigma = np.array([np.linalg.inv(sigma[:,:,l]) for l in range(bin_l) ])   
    
    
    
    
def log_prior(theta):
    if (-0.1 < theta[0] < 0.1 and -0.1 < theta[1] < 0.1 and 1.075 < theta[2] < 3.225 and 0.01 < theta[3] < 0.05 and 0.13 < theta[4] < 0.4 and 0.4 < theta[5] < 1.25 and 0.01 < theta[6] < 0.06 and 0.3 < theta[7] < 0.9 and 0.5 < theta[8] < 1.5 and -1.5 < theta[9] < -0.5 and -0.5 < theta[10] < 0.5):
        return 0.0
    return -np.inf


def lnl(theta): 
    cosmo = ccl.Cosmology(Omega_c=theta[4],
                      Omega_b = theta[6],
                      h=theta[7],
                      n_s=theta[8],
                      sigma8=theta[5],
                      w0 = theta[9],
                      wa = theta[10],
                      transfer_function='boltzmann_camb',
                      extra_parameters = {"camb": {"dark_energy_model": "ppf"}})
    D= ccl.growth_factor(cosmo, 1./(1+z_euclid))
    rho_m = ccl.physical_constants.RHO_CRITICAL * (cosmo['Omega_c']+cosmo['Omega_b'])
    Az_red =  theta[2]* C_1 * (rho_m / D)*((1+z_euclid)/(1+z_pivot))**theta[0]
    Az_blue = 5.*theta[3]* C_1 * (rho_m / (D**2))*((1+z_euclid)/(1+z_pivot))**theta[1]
    zeros_l = np.zeros(bin_l)
    clbtracer_eu_blue = np.array([ ccl.WeakLensingTracer(cosmo, dndz=(z_euclid, nz_blue[i,:]), ia_bias=(z_euclid, Az_blue), use_A_ia=False) for i in range(bin_euclid)])
    clbtracer_eu_red = np.array([ ccl.WeakLensingTracer(cosmo, dndz=(z_euclid, nz_red[i,:]), ia_bias=(z_euclid, Az_red), use_A_ia=False) for i in range(bin_euclid)])
    clbtracer = np.concatenate((clbtracer_eu_blue,clbtracer_eu_red))

    clij_smart_mod = np.array([ [
        ccl.angular_cl(cosmo, clbtracer[i], clbtracer[j], ell)
        if j<=i else np.zeros(bin_l)
        for j in range(bin_euclid*2) ] for i in range(bin_euclid*2) ]).T
    clij_smart_mod += np.triu(clij_smart_mod, 1).transpose(0, 2, 1)
    # Cl flattening
    clij_smart_flat_mod = np.array([[clij_smart_mod[l,i,j] for i in range(bin_euclid*2) for j in range(i,bin_euclid*2) ]for l in range(bin_l)])
    diffij_smart_flat = clij_smart_flat_mod - clij_smart_flat
    
    likehood = np.array([np.linalg.multi_dot([diffij_smart_flat[l,:],inv_sigma[l,:,:],diffij_smart_flat[l,:]]) for l in range(bin_l)])
    #print("something")
    return -np.sum(likehood)
    
    
    
    
    
def lnprob(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnl(theta) 
    
    
    
    
    
    
    
    
    
    
    
    
# Initialize the walkers

nwalkers = 32
ndim = 11
max_n = 100000
np.random.seed()
initial = [np.array([eta_red, eta_blue, A_red, A_blue, Oc , s8, Omega_b, h, n_s, w0, wa]) + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]


filename = "5paramtry.h5"
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers, ndim)



with Pool() as pool:
    
    move = emcee.moves.StretchMove()
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, backend=backend, pool=pool, moves=move)


    # We'll track how the average autocorrelation time estimate changes
    index = 0
    autocorr = np.empty(max_n)

    # This will be useful to testing convergence
    old_tau = np.inf

    # Now we'll sample for up to max_n steps
    for sample in sampler.sample(initial, iterations=max_n, progress=True):
        
        
        print("Iteration:", sampler.iteration)
        
        
        # Only check convergence every 100 steps
        if sampler.iteration % 100:
            continue

        # Compute the autocorrelation time so far
        # Using tol=0 means that we'll always get an estimate even
        # if it isn't trustworthy
        tau = sampler.get_autocorr_time(tol=0)
        autocorr[index] = np.mean(tau)
        index += 1
        
        print("Autocorrelation Time:", tau)
        

        # Check convergence
        converged = np.all(tau * 100 < sampler.iteration)
        converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
        if converged:
            
            print("Converged!")
            
            break
        old_tau = tau    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
