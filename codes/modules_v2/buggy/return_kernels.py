import healpy as h
import numpy as np
import euler as euler
from scipy.interpolate import interp1d

def return_qu2eb_kernel_convolve(nside,lon,lat,theta_cutoff,theta,rad_ker):
    fn_rad_ker=interp1d(np.cos(theta)[::-1],rad_ker[::-1],assume_sorted=True,kind="cubic",bounds_error=False,fill_value=0.0)
    npix=h.nside2npix(nside) ; domega=4.*np.pi/float(npix)

    cpix=h.ang2pix(nside,theta,phi,lonlat=True)
    cosbeta,cos2gamma,sin2gamma,spix=euler.fn_euler_trig2gamma(nside,cpix,theta_cutoff,inclusive=False,fact=4)
    rker=np.zeros(npix,"double") ; rker[spix]= cos2gamma*gbeta
    iker=np.zeros(npix,"double") ; iker[spix]= sin2gamma*gbeta
    return[rker,iker]

def return_qu2eb_kernel_radiate(nside,lon,lat,theta_cutoff,theta,rad_ker):
    fn_rad_ker=interp1d(np.cos(theta)[::-1],rad_ker[::-1],assume_sorted=True,kind="cubic",bounds_error=False,fill_value=0.0)
    npix=h.nside2npix(nside) ; domega=4.*np.pi/float(npix)

    cpix=h.ang2pix(nside,theta,phi,lonlat=True)
    cosbeta,cos2alpha,sin2alpha,spix=euler.fn_euler_trig2alpha(nside,cpix,theta_cutoff,inclusive=False,fact=4)

    rker=np.zeros(npix,"double") ; rker[spix]=  cos2alpha*gbeta
    iker=np.zeros(npix,"double") ; iker[spix]= -sin2alpha*gbeta
    return[rker,iker]

def return_I_kernel(nside,lon,lat,theta_cutoff,theta,rad_ker_i):
    fn_rad_ker_i=interp1d(np.cos(theta)[::-1],rad_ker_i[::-1],assume_sorted=True,kind="cubic",bounds_error=False,fill_value=0.0)
    npix=h.nside2npix(nside) ; domega=4.*np.pi/float(npix)
    
    cpix=h.ang2pix(nside,theta,phi,lonlat=True)
    cosbeta,c2apg,s2apg,spix=euler.fn_euler_trig2_alpha_plus_gamma(nside,cpix,theta_cutoff,inclusive=True,fact=4)

    rker=np.zeros(npix,"double") ; rker[spix]= c2apg*gbeta
    iker=np.zeros(npix,"double") ; iker[spix]= s2apg*gbeta
    
    return[rker,iker]

def return_D_kernel(nside,lon,lat,theta_cutoff,theta,rad_ker_d):
    fn_rad_ker_i=interp1d(np.cos(theta)[::-1],rad_ker_d[::-1],assume_sorted=True,kind="cubic",bounds_error=False,fill_value=0.0)
    npix=h.nside2npix(nside) ; domega=4.*np.pi/float(npix)
    
    cpix=h.ang2pix(nside,theta,phi,lonlat=True)
    cosbeta,c2amg,s2amg,spix=euler.fn_euler_trig2_alpha_minus_gamma(nside,cpix,theta_cutoff,inclusive=True,fact=4)

    rker=np.zeros(npix,"double") ; rker[spix]= c2amg*gbeta
    iker=np.zeros(npix,"double") ; iker[spix]= s2amg*gbeta
    
    return[rker,iker]

