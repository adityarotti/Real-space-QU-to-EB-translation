import healpy as h
import numpy as np
import euler as euler
from scipy.interpolate import interp1d

def convert_qu2eb_obsolete(q,u,theta_cutoff,theta,rad_ker):
	fn_rad_ker=interp1d(np.cos(theta)[::-1],rad_ker[::-1],assume_sorted=True,kind="cubic",bounds_error=False,fill_value=0.0)
	nside=h.get_nside(q) ; npix=h.nside2npix(nside) ; domega=4.*np.pi/float(npix)
	e=np.zeros(npix,"double")
	b=np.zeros(npix,"double")
	for i in range(npix):
		cosbeta,cos2alpha,sin2alpha,spix=euler.fn_s2euler_alpha(nside,i,theta_cutoff,inclusive=False,fact=4)
		gbeta=fn_rad_ker(cosbeta)
		e[i] = (np.dot(-cos2alpha*gbeta,q[spix]) + -np.dot(sin2alpha*gbeta,u[spix]))*domega
		b[i] = (np.dot(sin2alpha*gbeta,q[spix]) + np.dot(-cos2alpha*gbeta,u[spix]))*domega
	return [np.zeros(npix),e,b]


def convert_qu2eb_integrate(q,u,theta_cutoff,theta,rad_ker):
	fn_rad_ker=interp1d(np.cos(theta)[::-1],rad_ker[::-1],assume_sorted=True,kind="cubic",bounds_error=False,fill_value=0.0)
	nside=h.get_nside(q) ; npix=h.nside2npix(nside) ; domega=4.*np.pi/float(npix)
	e=np.zeros(npix,"double")
	b=np.zeros(npix,"double")
	for i in range(npix):
		cosbeta,cos2gamma,sin2gamma,spix=euler.fn_s2euler_gamma(nside,i,theta_cutoff,inclusive=False,fact=4)
		gbeta=fn_rad_ker(cosbeta)
		e[i] = (np.dot(-cos2gamma*gbeta,q[spix]) + np.dot(sin2gamma*gbeta,u[spix]))*domega
		b[i] = (np.dot(-sin2gamma*gbeta,q[spix]) + np.dot(-cos2gamma*gbeta,u[spix]))*domega
	return [np.zeros(npix),e,b]



def convert_qu2eb_radiate(q,u,theta_cutoff,theta,rad_ker):
	fn_rad_ker=interp1d(np.cos(theta)[::-1],rad_ker[::-1],assume_sorted=True,kind="cubic",bounds_error=False,fill_value=0.0)
	nside=h.get_nside(q) ; npix=h.nside2npix(nside) ; domega=4.*np.pi/float(npix)
	e=np.zeros(npix,"double")
	b=np.zeros(npix,"double")
	for i in range(npix):
		cosbeta,cos2alpha,sin2alpha,spix=euler.fn_s2euler_alpha(nside,i,theta_cutoff,inclusive=False,fact=4)
		gbeta=fn_rad_ker(cosbeta)
		cafb=cos2alpha*gbeta ; safb=sin2alpha*gbeta
		e[spix] = e[spix] + (-q[i]*cafb - u[i]*safb)*domega
		b[spix] = b[spix] + (q[i]*safb - u[i]*cafb)*domega
	return [np.zeros(npix),e,b]
