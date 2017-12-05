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

def convert_qu2_equ_bqu(q,u,theta_cutoff,theta,rad_ker_i,rad_ker_d):
	fn_rad_ker_i=interp1d(theta,rad_ker_i,assume_sorted=True,kind="cubic",bounds_error=False,fill_value=0.0)
	fn_rad_ker_d=interp1d(theta,rad_ker_d,assume_sorted=True,kind="cubic",bounds_error=False,fill_value=0.0)
	nside=h.get_nside(q) ; npix=h.nside2npix(nside) ; domega=4.*np.pi/float(npix)

	equ=[np.zeros(npix,"double"),np.zeros(npix,"double")]
	bqu=[np.zeros(npix,"double"),np.zeros(npix,"double")]
	
	for i in range(npix):
		beta,alpha,gamma,spix=return_euler_angles(nside,i,theta_cutoff,inclusive=True,fact=4)
		Ir=np.cos(2.*(alpha+gamma))*fn_rad_ker_i(beta)
		Ii=np.sin(2.*(alpha+gamma))*fn_rad_ker_i(beta)
		Dr=np.cos(2.*(alpha-gamma))*fn_rad_ker_d(beta)
		Di=np.sin(2.*(alpha-gamma))*fn_rad_ker_d(beta)

		equ[0][i]=(np.dot(Ir+Dr,q[spix]) + np.dot(Ii-Di,u[spix]))*domega*0.5
		equ[1][i]=(np.dot(-Ii-Di,q[spix]) + np.dot(Ir-Dr,u[spix]))*domega*0.5

		bqu[0][i]=(np.dot(Ir-Dr,q[spix]) + np.dot(Ii+Di,u[spix]))*domega*0.5
		bqu[1][i]=(np.dot(-Ii+Di,q[spix]) + np.dot(Ir+Dr,u[spix]))*domega*0.5

	return [np.zeros(npix),equ[0],equ[1]],[np.zeros(npix),bqu[0],bqu[1]]
			
	
