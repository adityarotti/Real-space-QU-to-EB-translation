import healpy as h
import numpy as np
import euler as euler
from scipy.interpolate import interp1d

# Convolution method to generate E and B mode maps.
def convert_eb2qu_convolve(e,b,theta_cutoff,theta,rad_ker,mask=[]):
	'''
	   This function uses E and B maps in reggions defined by the radial cutoff, irrespective of the mask.\n
	   It computes Q and U maps only inside the mask\n If you want it to compute only on masked E/B maps then just pass masked maps.
	'''
	fn_rad_ker=interp1d(theta,rad_ker,assume_sorted=True,kind="cubic",bounds_error=False,fill_value=0.0)
	nside=h.get_nside(e) ; npix=h.nside2npix(nside) ; domega=4.*np.pi/float(npix)

	if np.size(mask)==0:
		pix_list=np.arange(npix)
	else:
		pix_list=np.nonzero(mask)[0]

	q=np.zeros(npix,"double")
	u=np.zeros(npix,"double")
	for i in pix_list:
		beta,alpha,gamma,spix=euler.return_euler_angles(nside,i,theta_cutoff,inclusive=False,fact=4)
		gbeta=fn_rad_ker(beta)
		c2afb=np.cos(2.*alpha)*gbeta ; s2afb=np.sin(2.*alpha)*gbeta
		q[i] = (np.dot(-c2afb,e[spix]) + np.dot( s2afb,b[spix]))*domega
		u[i] = (np.dot(-s2afb,e[spix]) + np.dot(-c2afb,b[spix]))*domega

	return [np.zeros(npix),q,u]



# Radiating method to generate Q and U mode maps.
def convert_eb2qu_radiate(e,b,theta_cutoff,theta,rad_ker,mask=[]):
	'''This function only uses E and B maps inside the mask provided. It computes Q and U maps in regions defined by the radial cutoff'''
	fn_rad_ker=interp1d(theta,rad_ker,assume_sorted=True,kind="cubic",bounds_error=False,fill_value=0.0)
	nside=h.get_nside(e) ; npix=h.nside2npix(nside) ; domega=4.*np.pi/float(npix)

	if np.size(mask)==0:
		pix_list=np.arange(npix)
	else:
		pix_list=np.nonzero(mask)[0]

	q=np.zeros(npix,"double")
	u=np.zeros(npix,"double")

	for i in pix_list:
		beta,alpha,gamma,spix=euler.return_euler_angles(nside,i,theta_cutoff,inclusive=False,fact=4)
		gbeta=fn_rad_ker(beta)
		c2gfb=np.cos(2.*gamma)*gbeta ; s2gfb=np.sin(2.*gamma)*gbeta
		q[spix] = q[spix] + (-e[i]*c2gfb - b[i]*s2gfb)*domega
		u[spix] = u[spix] + ( e[i]*s2gfb - b[i]*c2gfb)*domega

	return [np.zeros(npix),q,u]
