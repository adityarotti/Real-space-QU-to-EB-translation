import healpy as h
import numpy as np
import euler as euler
from scipy.interpolate import interp1d

# Convolution method to generate E and B mode maps.
def convert_qu2eb_convolve(q,u,theta_cutoff,theta,rad_ker,mask=[]):
	'''
	   This function uses Q and U maps in regions defined by the radial cutoff, irrespective of the mask.\n
	   It computes E and B maps only inside the mask\n If you want it to compute only on masked Q/U maps then just pass masked maps.
	'''
	fn_rad_ker=interp1d(np.cos(theta)[::-1],rad_ker[::-1],assume_sorted=True,kind="cubic",bounds_error=False,fill_value=0.0)
	nside=h.get_nside(q) ; npix=h.nside2npix(nside) ; domega=4.*np.pi/float(npix)

	if np.size(mask)==0:
		pix_list=np.arange(npix)
	else:
		pix_list=np.nonzero(mask)[0]

	e=np.zeros(npix,"double")
	b=np.zeros(npix,"double")
	for i in pix_list:
		cosbeta,cos2gamma,sin2gamma,spix=euler.fn_euler_trig2gamma(nside,i,theta_cutoff,inclusive=False,fact=4)
		gbeta=fn_rad_ker(cosbeta)
		c2gfb=cos2gamma*gbeta ; s2gfb=sin2gamma*gbeta
		e[i] = (np.dot(-c2gfb,q[spix]) + np.dot( s2gfb,u[spix]))*domega
		b[i] = (np.dot(-s2gfb,q[spix]) + np.dot(-c2gfb,u[spix]))*domega
	return [np.zeros(npix),e,b]



# Radiating method to generate E and B mode maps.
def convert_qu2eb_radiate(q,u,theta_cutoff,theta,rad_ker,mask=[]):
	'''This function only uses Q and U maps inside the mask provided. It computes E and B maps in regions defined by the radial cutoff'''
	fn_rad_ker=interp1d(np.cos(theta)[::-1],rad_ker[::-1],assume_sorted=True,kind="cubic",bounds_error=False,fill_value=0.0)
	nside=h.get_nside(q) ; npix=h.nside2npix(nside) ; domega=4.*np.pi/float(npix)

	if np.size(mask)==0:
		pix_list=np.arange(npix)
	else:
		pix_list=np.nonzero(mask)[0]

	e=np.zeros(npix,"double")
	b=np.zeros(npix,"double")

	for i in pix_list:
		cosbeta,cos2alpha,sin2alpha,spix=euler.fn_euler_trig2alpha(nside,i,theta_cutoff,inclusive=False,fact=4)
		gbeta=fn_rad_ker(cosbeta)
		c2afb=cos2alpha*gbeta ; s2afb=sin2alpha*gbeta
		e[spix] = e[spix] + (-q[i]*c2afb - u[i]*s2afb)*domega
		b[spix] = b[spix] + ( q[i]*s2afb - u[i]*c2afb)*domega

	return [np.zeros(npix),e,b]
    
