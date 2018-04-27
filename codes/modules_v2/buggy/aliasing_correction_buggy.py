import healpy as h
import numpy as np
import euler as euler
from scipy.interpolate import interp1d

# Convolution method to convolve with a band limited delta function
def delta_convolve(q,u,theta_cutoff,theta,rad_ker_i,mask=[]):
	fn_rad_ker_i=interp1d(np.cos(theta)[::-1],rad_ker_i[::-1],assume_sorted=True,kind="cubic",bounds_error=False,fill_value=0.0)
	nside=h.get_nside(q) ; npix=h.nside2npix(nside) ; domega=4.*np.pi/float(npix)

	hq=np.zeros(npix,"double")
	hu=np.zeros(npix,"double")
	
	if np.size(mask)==0:
		pix_list=np.arange(npix)
	else:
		pix_list=np.nonzero(mask)[0]

	# The pixel i is where EQU and BQU are being evaluated.	
	for i in pix_list:
		cosbeta,c2apg,s2apg,spix=euler.fn_euler_trig2_alpha_plus_gamma(nside,i,theta_cutoff,inclusive=True,fact=4)
		gbeta=fn_rad_ker_i(cosbeta)
		Ir=c2apg*gbeta
		Ii=s2apg*gbeta

		hq[i]=(np.dot(Ir,q[spix]) + np.dot(-Ii,u[spix]))*domega
		hu[i]=(np.dot(Ii,q[spix]) + np.dot( Ir,u[spix]))*domega

	return hq,hu

def correct_aliasing_convolve(tq,tu,theta_cutoff,theta,rad_ker_i,iter=3,mask=[]):
	npix=np.size(tq)
	cq=np.zeros(npix,"double") ; cu=np.zeros(npix,"double")	
	for i in range(iter):
		hq,hu=delta_convolve(tq-cq,tu-cu,theta_cutoff,theta,rad_ker_i,mask)
		cq=cq + (hq-tq) ; cu=cu + (hu-tu)

	return [np.zeros(npix),tq-cq,tu-cu]

# Radiation method to convolve with a band limited delta function
def delta_radiate(q,u,theta_cutoff,theta,rad_ker_i,mask=[]):
	fn_rad_ker_i=interp1d(np.cos(theta)[::-1],rad_ker_i[::-1],assume_sorted=True,kind="cubic",bounds_error=False,fill_value=0.0)
	nside=h.get_nside(q) ; npix=h.nside2npix(nside) ; domega=4.*np.pi/float(npix)

	hq=np.zeros(npix,"double")
	hu=np.zeros(npix,"double")
	
	if np.size(mask)==0:
		pix_list=np.arange(npix)
	else:
		pix_list=np.nonzero(mask)[0]

	# Here the pixel i is where the Stokes Q/U map are being accessed.
        # We radiate out the delta function to the surrounding pixels. In infinite band limit,
        # this only contributed to the local pixel.	
	for i in pix_list:
		cosbeta,c2apg,s2apg,spix=euler.fn_euler_trig2_alpha_plus_gamma(nside,i,theta_cutoff,inclusive=True,fact=4)
		gbeta=fn_rad_ker_i(cosbeta)
		Ir=c2apg*gbeta
		Ii=s2apg*gbeta

		hq[spix]=hq[spix] + ( q[i]*Ir + u[i]*Ii)*domega
		hu[spix]=hu[spix] + (-q[i]*Ii + u[i]*Ir)*domega

	return hq,hu
			

def correct_aliasing_radiate(tq,tu,theta_cutoff,theta,rad_ker_i,iter=3,mask=[]):
	npix=np.size(tq)
	cq=np.zeros(npix,"double") ; cu=np.zeros(npix,"double")	
	for i in range(iter):
		hq,hu=delta_radiate(tq-cq,tu-cu,theta_cutoff,theta,rad_ker_i,mask)
		cq=cq + (hq-tq) ; cu=cu + (hu-tu)

	return [np.zeros(npix),tq-cq,tu-cu]
