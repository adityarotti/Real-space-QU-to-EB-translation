import healpy as h
import numpy as np

def return_disc_mask(discsize,apow,nside,cpix):
    mask=np.zeros(h.nside2npix(nside),float)
    # The coordinate of the pixel center updated to that in the mask resolution
    theta0=h.pix2ang(nside,cpix)[0] ; phi0=h.pix2ang(nside,cpix)[1]

    pixnum=np.arange(h.nside2npix(nside))
    theta=h.pix2ang(nside,pixnum)[0] ; phi=h.pix2ang(nside,pixnum)[1]
    dotprod=np.sin(theta)*np.sin(theta0)*np.cos(phi-phi0)+np.cos(theta)*np.cos(theta0)
    dtheta=np.arccos(dotprod)*180./np.pi

    for i in range(h.nside2npix(nside)):
        if dtheta[i] <= (discsize-apow):
            mask[i]=1.
        elif dtheta[i] > (discsize-apow) and dtheta[i] <= discsize:
            mask[i]=np.cos((dtheta[i]-(discsize-apow))*np.pi/(2.*apow))**2.
            
    return mask

def return_local_eb(tqu,discsize,nside,lmax,lmin=2.,apow=2.,return_equ_bqu=False):
	
	npix=h.nside2npix(nside)
	rteb=[np.zeros(npix,float),np.zeros(npix,float)]

	if return_equ_bqu:
		requ=[np.zeros(npix,float),np.zeros(npix,float)]
		rbqu=[np.zeros(npix,float),np.zeros(npix,float)]
	
		for i in range(npix):
			mask=return_disc_mask(discsize,apow,nside,i)
			temp_alm=h.map2alm(tqu*mask,lmax=lmax,pol=True)
			temp_teb=h.alm2map(temp_alm,nside,pol=False,verbose=False)
			temp_equ=h.alm2map([temp_alm[0],temp_alm[1],temp_alm[2]*0.],nside,pol=True,verbose=False)
			temp_bqu=h.alm2map([temp_alm[0],temp_alm[1]*0.,temp_alm[2]],nside,pol=True,verbose=False)
			rteb[0][i]=temp_teb[1][i] ; rteb[1][i]=temp_teb[2][i]
			requ[0][i]=temp_equ[1][i] ; requ[1][i]=temp_equ[2][i]
			rbqu[0][i]=temp_bqu[1][i] ; rbqu[1][i]=temp_bqu[2][i]

		return [np.zeros(npix,float),rteb[0],rteb[1]],[np.zeros(npix,float),requ[0],requ[1]],[np.zeros(npix,float),rbqu[0],rbqu[1]]
	else:
		for i in range(npix):
			mask=return_disc_mask(discsize,apow,nside,i)
			temp_alm=h.map2alm(tqu*mask,lmax=lmax,pol=True)
			temp_teb=h.alm2map(temp_alm,nside,pol=False,verbose=False)
			rteb[0][i]=temp_teb[1][i] ; rteb[1][i]=temp_teb[2][i]

		return [np.zeros(npix,float),rteb[0],rteb[1]],0,0
