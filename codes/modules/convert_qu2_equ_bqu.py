import healpy as h
import numpy as np
import euler as euler
from scipy.interpolate import interp1d

def convert_qu2_equ_bqu(q,u,theta_cutoff,theta,rad_ker_i,rad_ker_d):
	fn_rad_ker_i=interp1d(np.cos(theta)[::-1],rad_ker_i[::-1],assume_sorted=True,kind="cubic",bounds_error=False,fill_value=0.0)
	fn_rad_ker_d=interp1d(np.cos(theta)[::-1],rad_ker_d[::-1],assume_sorted=True,kind="cubic",bounds_error=False,fill_value=0.0)
	nside=h.get_nside(q) ; npix=h.nside2npix(nside) ; domega=4.*np.pi/float(npix)

	equ=[np.zeros(npix,"double"),np.zeros(npix,"double")]
	bqu=[np.zeros(npix,"double"),np.zeros(npix,"double")]
	
	for i in range(npix):
		cosbeta,c2apg,s2apg,c2amg,s2amg,spix=euler.fn_s2euler_alpha_gamma(nside,i,theta_cutoff,inclusive=False,fact=4)

		Ir=c2apg*fn_rad_ker_i(cosbeta)
		Ii=s2apg*fn_rad_ker_i(cosbeta)
		Dr=c2amg*fn_rad_ker_d(cosbeta)
		Di=s2amg*fn_rad_ker_d(cosbeta)

		equ[0][i]=(np.dot(Ir+Dr,q[spix]) + np.dot(Ii-Di,u[spix]))*domega*0.5
		equ[1][i]=(np.dot(-Ii-Di,q[spix]) + np.dot(Ir-Dr,u[spix]))*domega*0.5

		bqu[0][i]=(np.dot(Ir-Dr,q[spix]) + np.dot(Ii+Di,u[spix]))*domega*0.5
		bqu[1][i]=(np.dot(-Ii+Di,q[spix]) + np.dot(Ir+Dr,u[spix]))*domega*0.5

	return [np.zeros(npix),equ[0],equ[1]],[np.zeros(npix),bqu[0],bqu[1]]
			
	
