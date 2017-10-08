import healpy as h
import numpy as np

def return_kernel_qu2eb(nside,lmax,pixi,thetai=0,phii=0,use_coord=False,norm="Int",sample_nside=-1):
    
    if sample_nside==-1:
        sample_nside=nside

    if norm=="Int":
        norm=0.5*(4.*np.pi)/h.nside2npix(nside)
    else:
        norm=1.

    if use_coord:
        pixi=h.ang2pix(nside,thetai,phii)
        thetai,phii=h.pix2ang(nside,pixi)
    else:
        thetai,phii=h.pix2ang(nside,pixi)

    almsize=h.Alm.getsize(lmax=lmax,mmax=lmax)
    almr0=np.zeros(almsize,complex)
    almi0=np.zeros(almsize,complex)

    for i in range(almsize):
        l,m=h.Alm.getlm(lmax,i)
        if l>1 and m==2:
            almr0[i]=complex(1.0,0.0)*np.sqrt((2.*l+1.)/(4.*np.pi))*norm
            almi0[i]=complex(0.0,-1.0)*np.sqrt((2.*l+1.)/(4.*np.pi))*norm

    h.rotate_alm(almr0,0.0,thetai,phii)
    h.rotate_alm(almi0,0.0,thetai,phii)

    mr=h.alm2map(almr0,sample_nside,lmax=lmax,mmax=lmax,verbose=False)
    mi=h.alm2map(almi0,sample_nside,lmax=lmax,mmax=lmax,verbose=False)

    return mr,mi
    

def return_kernel_I(nside,lmax,pixi=-1,thetai=0,phii=0,use_coord=False,norm="Int",sample_nside=-1):
    '''
    If pixi=-1 then the alms correspond to the kernel exactly at the pole. Else the alms are rotated to 
    allign with the specified coordinate or the coordinate of the pixel.
    '''
    if sample_nside==-1:
        sample_nside=nside

    if norm=="Int":
        norm=0.5*(4.*np.pi)/h.nside2npix(nside)
    else:
        norm=1.

    if use_coord:
        pixi=h.ang2pix(nside,thetai,phii)
        thetai,phii=h.pix2ang(nside,pixi)
    else:
        thetai,phii=h.pix2ang(nside,pixi)

    almsize=h.Alm.getsize(lmax=lmax,mmax=lmax)
    alm0=[np.zeros(almsize,complex),np.zeros(almsize,complex),np.zeros(almsize,complex)]

    for i in range(almsize):
        l,m=h.Alm.getlm(lmax,i)
        if l>1 and m==2:
            alm0[1][i]=0.5*complex(-1.0,0.0)*np.sqrt((2.*l+1.)/(4.*np.pi))*norm
            alm0[2][i]=0.5*complex(0.0,-1.0)*np.sqrt((2.*l+1.)/(4.*np.pi))*norm

    if pixi!=-1:
        h.rotate_alm(alm0,0.0,thetai,phii)

    mqu=h.alm2map(alm0,sample_nside,lmax=lmax,mmax=lmax,pol=True,verbose=False)

    return mqu[1],-mqu[2]

def return_kernel_J(nside,lmax,pixi=-1,thetai=0,phii=0,use_coord=False,norm="Int",sample_nside=-1):
    '''
    If pixi=-1 then the alms correspond to the kernel exactly at the pole. Else the alms are rotated to 
    allign with the specified coordinate or the coordinate of the pixel.
    '''
    if sample_nside==-1:
        sample_nside=nside

    if norm=="Int":
        norm=0.5*(4.*np.pi)/h.nside2npix(nside)
    else:
        norm=1.

    if use_coord:
        pixi=h.ang2pix(nside,thetai,phii)
        thetai,phii=h.pix2ang(nside,pixi)
    else:
        thetai,phii=h.pix2ang(nside,pixi)

    almsize=h.Alm.getsize(lmax=lmax,mmax=lmax)
    alm0=[np.zeros(almsize,complex),np.zeros(almsize,complex),np.zeros(almsize,complex)]

    for i in range(almsize):
        l,m=h.Alm.getlm(lmax,i)
        if l>1 and m==2:
            alm0[1][i]=0.5*complex(-1.0,0.0)*np.sqrt((2.*l+1.)/(4.*np.pi))*norm
            alm0[2][i]=0.5*complex(0.0,1.0)*np.sqrt((2.*l+1.)/(4.*np.pi))*norm

    if pixi!=-1:
        h.rotate_alm(alm0,0.0,thetai,phii)

    mqu=h.alm2map(alm0,sample_nside,lmax=lmax,mmax=lmax,pol=True,verbose=False)

    return mqu[1],-mqu[2]

