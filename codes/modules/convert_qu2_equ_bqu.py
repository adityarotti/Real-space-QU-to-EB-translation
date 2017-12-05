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

def convert_qu2_equ_bqu_integrate(q,u,theta_cutoff,theta,rad_ker_d,cq=[],cu=[],mask=[]):
	'''Pass alias corrected TQU map to this routine.'''
	'''The delta function should act like an identity operator, but it does not on the Healpix pixelization'''
	'''The delta term is thought to operate on the corrected TQU to return true TQU, but thats not the case as it returns TQU with error of order iter+1 order \n
	where iter refers to the number of correction terms added to the tqu maps passed to this routine.'''
	fn_rad_ker_d=interp1d(theta,rad_ker_d,assume_sorted=True,kind="cubic",bounds_error=False,fill_value=0.0)
	nside=h.get_nside(q) ; npix=h.nside2npix(nside) ; domega=4.*np.pi/float(npix)

	if np.size(mask)==0:
		pix_list=np.arange(npix)
	else:
		pix_list=np.nonzero(mask)[0]


	dqu=[np.zeros(npix,"double"),np.zeros(npix,"double")]
	
	if np.size(cq)!=0 and np.size(cu)!=0:
		for i in pix_list:
			beta,alpha,gamma,spix=return_euler_angles(nside,i,theta_cutoff,inclusive=True,fact=4)
			Dr=np.cos(2.*(alpha-gamma))*fn_rad_ker_d(beta)
			Di=np.sin(2.*(alpha-gamma))*fn_rad_ker_d(beta)

			dqu[0][i]=(np.dot(+Dr,cq[spix]) + np.dot(-Di,cu[spix]))*domega
			dqu[1][i]=(np.dot(-Di,cq[spix]) + np.dot(-Dr,cu[spix]))*domega
	else:
		for i in pix_list:
			beta,alpha,gamma,spix=return_euler_angles(nside,i,theta_cutoff,inclusive=True,fact=4)
			Dr=np.cos(2.*(alpha-gamma))*fn_rad_ker_d(beta)
			Di=np.sin(2.*(alpha-gamma))*fn_rad_ker_d(beta)

			dqu[0][i]=(np.dot(+Dr,q[spix]) + np.dot(-Di,u[spix]))*domega
			dqu[1][i]=(np.dot(-Di,q[spix]) + np.dot(-Dr,u[spix]))*domega


	return [np.zeros(npix),(q+dqu[0])*0.5,(u+dqu[1])*0.5],[np.zeros(npix),(q-dqu[0])*0.5,(u-dqu[1])*0.5]
			
def convert_qu2_equ_bqu_radiate(q,u,theta_cutoff,theta,rad_ker_d,cq=[],cu=[],mask=[]):
	'''Pass alias corrected TQU map to this routine.'''
	'''The delta function should act like an identity operator, but it does not on the Healpix pixelization'''
	'''The delta term is thought to operate on the corrected TQU to return true TQU, but thats not the case as it returns TQU with error of order iter+1 order \n
	where iter refers to the number of correction terms added to the tqu maps passed to this routine.'''
	fn_rad_ker_d=interp1d(theta,rad_ker_d,assume_sorted=True,kind="cubic",bounds_error=False,fill_value=0.0)
	nside=h.get_nside(q) ; npix=h.nside2npix(nside) ; domega=4.*np.pi/float(npix)

	if np.size(mask)==0:
		pix_list=np.arange(npix)
	else:
		pix_list=np.nonzero(mask)[0]


	dqu=[np.zeros(npix,"double"),np.zeros(npix,"double")]
	
	if np.size(cq)!=0 and np.size(cu)!=0:
		for i in pix_list:
			beta,alpha,gamma,spix=return_euler_angles(nside,i,theta_cutoff,inclusive=True,fact=4)
			Dr=np.cos(2.*(alpha-gamma))*fn_rad_ker_d(beta)
			Di=np.sin(2.*(alpha-gamma))*fn_rad_ker_d(beta)

			dqu[0][spix]=dqu[0][spix] + ( cq[i]*Dr - cu[i]*Di)*domega
			dqu[1][spix]=dqu[1][spix] + (-cq[i]*Di - cu[i]*Dr)*domega
	else:
		for i in pix_list:
			beta,alpha,gamma,spix=return_euler_angles(nside,i,theta_cutoff,inclusive=True,fact=4)
			Dr=np.cos(2.*(alpha-gamma))*fn_rad_ker_d(beta)
			Di=np.sin(2.*(alpha-gamma))*fn_rad_ker_d(beta)

			dqu[0][spix]=dqu[0][spix] + ( q[i]*Dr - u[i]*Di)*domega
			dqu[1][spix]=dqu[1][spix] + (-q[i]*Di - u[i]*Dr)*domega

	return [np.zeros(npix),(q+dqu[0])*0.5,(u+dqu[1])*0.5],[np.zeros(npix),(q-dqu[0])*0.5,(u-dqu[1])*0.5]
