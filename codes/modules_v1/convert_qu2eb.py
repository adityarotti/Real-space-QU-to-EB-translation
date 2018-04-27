import healpy as h
import numpy as np
import euler as euler
from scipy.interpolate import interp1d

# Integrating method to generate E and B mode maps.
def convert_qu2eb_integrate(q,u,theta_cutoff,theta,rad_ker):
	'''
	   This function uses Q and U maps in regions defined by the radial cutoff, irrespective of the mask.\n
	   It computes E and B maps only inside the mask\n If you want it to compute only on masked Q/U maps then just pass masked maps.
	'''
	fn_rad_ker=interp1d(np.cos(theta)[::-1],rad_ker[::-1],assume_sorted=True,kind="cubic",bounds_error=False,fill_value=0.0)
	nside=h.get_nside(q) ; npix=h.nside2npix(nside) ; domega=4.*np.pi/float(npix)
	e=np.zeros(npix,"double")
	b=np.zeros(npix,"double")
	for i in range(npix):
		cosbeta,cos2gamma,sin2gamma,spix=euler.fn_s2euler_gamma(nside,i,theta_cutoff,inclusive=False,fact=4)
		gbeta=fn_rad_ker(cosbeta)
		cgfb=cos2gamma*gbeta ; sgfb=sin2gamma*gbeta
		e[i] = (np.dot(-cgfb,q[spix]) + np.dot( sgfb,u[spix]))*domega
		b[i] = (np.dot(-sgfb,q[spix]) + np.dot(-cgfb,u[spix]))*domega
	return [np.zeros(npix),e,b]


def convert_qu2eb_integrate_masked(q,u,theta_cutoff,theta,rad_ker,mask=[]):
	'''
	   This function uses Q and U maps in reggions defined by the radial cutoff, irrespective of the mask.\n
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
		cosbeta,cos2gamma,sin2gamma,spix=euler.fn_s2euler_gamma(nside,i,theta_cutoff,inclusive=False,fact=4)
		gbeta=fn_rad_ker(cosbeta)
		cgfb=cos2gamma*gbeta ; sgfb=sin2gamma*gbeta
		e[i] = (np.dot(-cgfb,q[spix]) + np.dot( sgfb,u[spix]))*domega
		b[i] = (np.dot(-sgfb,q[spix]) + np.dot(-cgfb,u[spix]))*domega
	return [np.zeros(npix),e,b]



# Radiating method to generate E and B mode maps.
def convert_qu2eb_radiate(q,u,theta_cutoff,theta,rad_ker):
	'''This function only uses Q and U maps inside the mask provided. It computes E and B maps in regions defined by the radial cutoff'''
	fn_rad_ker=interp1d(np.cos(theta)[::-1],rad_ker[::-1],assume_sorted=True,kind="cubic",bounds_error=False,fill_value=0.0)
	nside=h.get_nside(q) ; npix=h.nside2npix(nside) ; domega=4.*np.pi/float(npix)
	e=np.zeros(npix,"double")
	b=np.zeros(npix,"double")
	for i in range(npix):
		cosbeta,cos2alpha,sin2alpha,spix=euler.fn_s2euler_alpha(nside,i,theta_cutoff,inclusive=False,fact=4)
		gbeta=fn_rad_ker(cosbeta)
		cafb=cos2alpha*gbeta ; safb=sin2alpha*gbeta
		e[spix] = e[spix] + (-cafb*q[i] + safb*u[i])*domega
		b[spix] = b[spix] + (-safb*q[i] - cafb*u[i])*domega
	return [np.zeros(npix),e,b]


def convert_qu2eb_radiate_masked(q,u,theta_cutoff,theta,rad_ker,mask=[]):
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
		cosbeta,cos2alpha,sin2alpha,spix=euler.fn_s2euler_alpha(nside,i,theta_cutoff,inclusive=False,fact=4)
		gbeta=fn_rad_ker(cosbeta)
		cafb=cos2alpha*gbeta ; safb=sin2alpha*gbeta
		e[spix] = e[spix] + (-q[i]*cafb + u[i]*safb)*domega
		b[spix] = b[spix] + (-q[i]*safb - u[i]*cafb)*domega
	return [np.zeros(npix),e,b]

###########################################################################################
# This function is wrong. Uses the radiating kernel to do the integration.
# Maybe useful to check where the flat sky approximation begins to fail.
def convert_qu2eb_obsolete(q,u,theta_cutoff,theta,rad_ker):
	fn_rad_ker=interp1d(np.cos(theta)[::-1],rad_ker[::-1],assume_sorted=True,kind="cubic",bounds_error=False,fill_value=0.0)
	nside=h.get_nside(q) ; npix=h.nside2npix(nside) ; domega=4.*np.pi/float(npix)
	e=np.zeros(npix,"double")
	b=np.zeros(npix,"double")
	for i in range(npix):
		cosbeta,cos2alpha,sin2alpha,spix=euler.fn_s2euler_alpha(nside,i,theta_cutoff,inclusive=False,fact=4)
		gbeta=fn_rad_ker(cosbeta)
		e[i] = (np.dot(-cos2alpha*gbeta,q[spix]) + np.dot(-sin2alpha*gbeta,u[spix]))*domega
		b[i] = (np.dot( sin2alpha*gbeta,q[spix]) + np.dot(-cos2alpha*gbeta,u[spix]))*domega
	return [np.zeros(npix),e,b]
###########################################################################################

