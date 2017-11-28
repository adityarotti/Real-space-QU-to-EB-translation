import healpy as h
import numpy as np
import euler as euler
from scipy.interpolate import interp1d

# Integrating method to generate E and B mode maps.
def convert_eb2qu_integrate(e,b,theta_cutoff,theta,rad_ker):
	'''
	   This function uses E and B maps in regions defined by the radial cutoff, irrespective of the mask.\n
	   It computes Q and U maps only inside the mask\n If you want it to compute only on masked E/B maps then just pass masked maps.
	'''
	fn_rad_ker=interp1d(np.cos(theta)[::-1],rad_ker[::-1],assume_sorted=True,kind="cubic",bounds_error=False,fill_value=0.0)
	nside=h.get_nside(e) ; npix=h.nside2npix(nside) ; domega=4.*np.pi/float(npix)
	q=np.zeros(npix,"double")
	u=np.zeros(npix,"double")
	for i in range(npix):
		cosbeta,cos2alpha,sin2alpha,spix=euler.fn_s2euler_alpha(nside,i,theta_cutoff,inclusive=False,fact=4)
		gbeta=fn_rad_ker(cosbeta)
		cafb=cos2alpha*gbeta ; safb=sin2alpha*gbeta
		q[i] = (np.dot(-cafb,e[spix]) + np.dot(-safb,b[spix]))*domega
		u[i] = (np.dot( safb,e[spix]) + np.dot(-cafb,b[spix]))*domega
	return [np.zeros(npix),q,u]


def convert_eb2qu_integrate_masked(e,b,theta_cutoff,theta,rad_ker,mask=[]):
	'''
	   This function uses E and B maps in reggions defined by the radial cutoff, irrespective of the mask.\n
	   It computes Q and U maps only inside the mask\n If you want it to compute only on masked E/B maps then just pass masked maps.
	'''
	fn_rad_ker=interp1d(np.cos(theta)[::-1],rad_ker[::-1],assume_sorted=True,kind="cubic",bounds_error=False,fill_value=0.0)
	nside=h.get_nside(e) ; npix=h.nside2npix(nside) ; domega=4.*np.pi/float(npix)

	if np.size(mask)==0:
		pix_list=np.arange(npix)
	else:
		pix_list=np.nonzero(mask)[0]

	q=np.zeros(npix,"double")
	u=np.zeros(npix,"double")
	for i in pix_list:
		cosbeta,cos2alpha,sin2alpha,spix=euler.fn_s2euler_alpha(nside,i,theta_cutoff,inclusive=False,fact=4)
		gbeta=fn_rad_ker(cosbeta)
		cafb=cos2alpha*gbeta ; safb=sin2alpha*gbeta
		q[i] = (np.dot(-cafb,e[spix]) + np.dot(-safb,b[spix]))*domega
		u[i] = (np.dot( safb,e[spix]) + np.dot(-cafb,b[spix]))*domega
	return [np.zeros(npix),q,u]



# Radiating method to generate Q and U mode maps.
def convert_eb2qu_radiate(e,b,theta_cutoff,theta,rad_ker):
	'''This function only uses E and B maps inside the mask provided. It computes Q and U maps in regions defined by the radial cutoff'''
	fn_rad_ker=interp1d(np.cos(theta)[::-1],rad_ker[::-1],assume_sorted=True,kind="cubic",bounds_error=False,fill_value=0.0)
	nside=h.get_nside(e) ; npix=h.nside2npix(nside) ; domega=4.*np.pi/float(npix)
	q=np.zeros(npix,"double")
	u=np.zeros(npix,"double")
	for i in range(npix):
		cosbeta,cos2gamma,sin2gamma,spix=euler.fn_s2euler_gamma(nside,i,theta_cutoff,inclusive=False,fact=4)
		gbeta=fn_rad_ker(cosbeta)
		cgfb=cos2gamma*gbeta ; sgfb=sin2gamma*gbeta
		q[spix] = q[spix] + (-cgfb*e[i] - sgfb*b[i])*domega
		u[spix] = u[spix] + ( sgfb*e[i] - cgfb*b[i])*domega
	return [np.zeros(npix),q,u]


def convert_eb2qu_radiate_masked(e,b,theta_cutoff,theta,rad_ker,mask=[]):
	'''This function only uses E and B maps inside the mask provided. It computes Q and U maps in regions defined by the radial cutoff'''
	fn_rad_ker=interp1d(np.cos(theta)[::-1],rad_ker[::-1],assume_sorted=True,kind="cubic",bounds_error=False,fill_value=0.0)
	nside=h.get_nside(e) ; npix=h.nside2npix(nside) ; domega=4.*np.pi/float(npix)

	if np.size(mask)==0:
		pix_list=np.arange(npix)
	else:
		pix_list=np.nonzero(mask)[0]

	q=np.zeros(npix,"double")
	u=np.zeros(npix,"double")

	for i in pix_list:
		cosbeta,cos2gamma,sin2gamma,spix=euler.fn_s2euler_gamma(nside,i,theta_cutoff,inclusive=False,fact=4)
		gbeta=fn_rad_ker(cosbeta)
		cgfb=cos2gamma*gbeta ; sgfb=sin2gamma*gbeta
		q[spix] = q[spix] + (-e[i]*cgfb - b[i]*sgfb)*domega
		u[spix] = u[spix] + ( e[i]*sgfb - b[i]*cgfb)*domega
	return [np.zeros(npix),q,u]

###########################################################################################
# This function is wrong. Uses the radiating kernel to do the integration.
# Maybe useful to check where the flat sky approximation begins to fail.
def convert_eb2qu_obsolete(e,b,theta_cutoff,theta,rad_ker):
	fn_rad_ker=interp1d(np.cos(theta)[::-1],rad_ker[::-1],assume_sorted=True,kind="cubic",bounds_error=False,fill_value=0.0)
	nside=h.get_nside(e) ; npix=h.nside2npix(nside) ; domega=4.*np.pi/float(npix)
	q=np.zeros(npix,"double")
	u=np.zeros(npix,"double")
	for i in range(npix):
		cosbeta,cos2alpha,sin2alpha,spix=euler.fn_s2euler_alpha(nside,i,theta_cutoff,inclusive=False,fact=4)
		gbeta=fn_rad_ker(cosbeta)
		q[i] = (np.dot(-cos2alpha*gbeta,e[spix]) + np.dot(-sin2alpha*gbeta,b[spix]))*domega
		u[i] = (np.dot( sin2alpha*gbeta,e[spix]) + np.dot(-cos2alpha*gbeta,b[spix]))*domega
	return [np.zeros(npix),e,b]
###########################################################################################

