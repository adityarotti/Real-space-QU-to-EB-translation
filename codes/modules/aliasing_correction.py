import healpy as h
import numpy as np
import euler as euler
from scipy.interpolate import interp1d

def return_euler_angles(nside,cpix,discsize=np.pi,inclusive=False,fact=4):
	'''
	This Euler angles function works for the qu2_equ_bqu function.
	The set of function here, match Healpix results to numerical accuracy.
	'''
	theta1,phi1=h.pix2ang(nside,cpix)
	v=h.pix2vec(nside,cpix)
	spix=h.query_disc(nside,v,discsize,inclusive=inclusive,fact=fact)
	theta2,phi2=h.pix2ang(nside,spix)

	cosbeta=np.sin(theta1)*np.sin(theta2)*np.cos(phi2-phi1)+np.cos(theta1)*np.cos(theta2) 
	cosbeta[cosbeta>1.]=1. ; cosbeta[cosbeta<-1.]=-1.
	beta=np.arccos(cosbeta)

	n = np.sin(theta1)*np.sin(theta2)*np.sin(phi1-phi2)

    	da = np.cos(theta1)*cosbeta - np.cos(theta2)
    	alpha=np.arctan2(n,da) ; alpha[beta==0]=0. ; alpha[abs(beta-np.pi)<1e-5]=0.

	dg = np.cos(theta2)*cosbeta - np.cos(theta1)
    	gamma=np.arctan2(n,dg) ; gamma[beta==0]=0. ; gamma[abs(beta-np.pi)<1e-5]=0.

	return beta,alpha,gamma,spix

def delta_convolve(q,u,theta_cutoff,theta,rad_ker_i,mask=[]):
	fn_rad_ker_i=interp1d(theta,rad_ker_i,assume_sorted=True,kind="cubic",bounds_error=False,fill_value=0.0)
	nside=h.get_nside(q) ; npix=h.nside2npix(nside) ; domega=4.*np.pi/float(npix)

	hq=np.zeros(npix,"double")
	hu=np.zeros(npix,"double")
	
	if np.size(mask)==0:
		pix_list=np.arange(npix)
	else:
		pix_list=np.nonzero(mask)[0]

		
	for i in pix_list:
		beta,alpha,gamma,spix=return_euler_angles(nside,i,theta_cutoff,inclusive=True,fact=4)
		Ir=np.cos(2.*(alpha+gamma))*fn_rad_ker_i(beta)
		Ii=np.sin(2.*(alpha+gamma))*fn_rad_ker_i(beta)

		hq[i]=(np.dot( Ir,q[spix]) + np.dot(Ii,u[spix]))*domega
		hu[i]=(np.dot(-Ii,q[spix]) + np.dot(Ir,u[spix]))*domega

	return hq,hu
			

def correct_aliasing(tq,tu,theta_cutoff,theta,rad_ker_i,iter=3,mask=[]):
	npix=np.size(tq)
	cq=np.zeros(npix,"double") ; cu=np.zeros(npix,"double")	
	for i in range(iter):
		hq,hu=delta_convolve(tq-cq,tu-cu,theta_cutoff,theta,rad_ker_i,mask)
		cq=cq + (hq-tq) ; cu=cu + (hu-tu)

	return [np.zeros(npix),tq-cq,tu-cu]
