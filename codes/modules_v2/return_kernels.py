import healpy as h
import numpy as np
import euler as euler
from scipy.interpolate import interp1d


def return_qu2eb_kernel_radiate(nside,lon,lat,theta_cutoff,theta,rad_ker,normalize=False):
    fn_rad_ker=interp1d(theta,rad_ker,assume_sorted=True,kind="cubic",bounds_error=False,fill_value=0.0)
    npix=h.nside2npix(nside) 

    domega=1.
    if normalize:
        domega=4.*np.pi/float(npix)
    
    cpix=h.ang2pix(nside,lon,lat,lonlat=True)

    beta,alpha,gamma,spix=euler.return_euler_angles(nside,cpix,theta_cutoff,inclusive=False,fact=4)
    
    gbeta=fn_rad_ker(beta)
    rker=np.zeros(npix,"double") ; rker[spix]= -np.cos(2.*alpha)*gbeta*domega
    iker=np.zeros(npix,"double") ; iker[spix]=  np.sin(2.*alpha)*gbeta*domega
    return cpix,[rker,iker]

def return_qu2eb_kernel_convolve(nside,lon,lat,theta_cutoff,theta,rad_ker,normalize=False):
    fn_rad_ker=interp1d(theta,rad_ker,assume_sorted=True,kind="cubic",bounds_error=False,fill_value=0.0)
    npix=h.nside2npix(nside) 

    domega=1.
    if normalize:
        domega=4.*np.pi/float(npix)

    cpix=h.ang2pix(nside,lon,lat,lonlat=True)

    beta,alpha,gamma,spix=euler.return_euler_angles(nside,cpix,theta_cutoff,inclusive=False,fact=4)

    gbeta=fn_rad_ker(beta)
    rker=np.zeros(npix,"double") ; rker[spix]= -np.cos(2.*gamma)*gbeta*domega
    iker=np.zeros(npix,"double") ; iker[spix]= -np.sin(2.*gamma)*gbeta*domega
    return cpix,[rker,iker]

def return_eb2qu_kernel_radiate(nside,lon,lat,theta_cutoff,theta,rad_ker,normalize=False):
    fn_rad_ker=interp1d(theta,rad_ker,assume_sorted=True,kind="cubic",bounds_error=False,fill_value=0.0)
    npix=h.nside2npix(nside) 

    domega=1.
    if normalize:
        domega=4.*np.pi/float(npix)

    cpix=h.ang2pix(nside,lon,lat,lonlat=True)

    beta,alpha,gamma,spix=euler.return_euler_angles(nside,cpix,theta_cutoff,inclusive=False,fact=4)
    
    gbeta=fn_rad_ker(beta)
    rker=np.zeros(npix,"double") ; rker[spix]= -np.cos(2.*gamma)*gbeta*domega
    iker=np.zeros(npix,"double") ; iker[spix]= np.sin(2.*gamma)*gbeta*domega
    return cpix,[rker,iker]

def return_I_kernel_radiate(nside,lon,lat,theta_cutoff,theta,rad_ker_i,normalize=False):
    fn_rad_ker_i=interp1d(theta,rad_ker_i,assume_sorted=True,kind="cubic",bounds_error=False,fill_value=0.0)
    npix=h.nside2npix(nside) 
    
    domega=1.
    if normalize:
        domega=4.*np.pi/float(npix)

    cpix=h.ang2pix(nside,lon,lat,lonlat=True)

    beta,alpha,gamma,spix=euler.return_euler_angles(nside,cpix,theta_cutoff,inclusive=True,fact=4)
    
    gbeta=fn_rad_ker_i(beta)
    rker=np.zeros(npix,"double") ; rker[spix]= np.cos(2.*(alpha+gamma))*gbeta*domega
    iker=np.zeros(npix,"double") ; iker[spix]= np.sin(2.*(alpha+gamma))*gbeta*domega
    return cpix,[rker,iker]

def return_D_kernel_radcon(nside,lon,lat,theta_cutoff,theta,rad_ker_d,normalize=False):
    fn_rad_ker_d=interp1d(theta,rad_ker_d,assume_sorted=True,kind="cubic",bounds_error=False,fill_value=0.0)
    npix=h.nside2npix(nside) 

    domega=1.
    if normalize:
        domega=4.*np.pi/float(npix)

    cpix=h.ang2pix(nside,lon,lat,lonlat=True)
    
    beta,alpha,gamma,spix=euler.return_euler_angles(nside,cpix,theta_cutoff,inclusive=True,fact=4)
    
    gbeta=fn_rad_ker_d(beta)
    rker=np.zeros(npix,"double") ; rker[spix]= np.cos(2.*(alpha-gamma))*gbeta*domega
    iker=np.zeros(npix,"double") ; iker[spix]= np.sin(2.*(alpha-gamma))*gbeta*domega
    return cpix,[rker,iker]

