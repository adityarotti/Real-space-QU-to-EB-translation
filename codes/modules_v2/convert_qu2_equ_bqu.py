import healpy as h
import numpy as np
import euler as euler
from scipy.interpolate import interp1d

# Convolution method to separate Q U into Stokes parameters corresponding to E and B modes respectively
def convert_qu2_equ_bqu_convolve(q,u,theta_cutoff,theta,rad_ker_d,cq=[],cu=[],mask=[]):
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
			beta,alpha,gamma,spix=euler.return_euler_angles(nside,i,theta_cutoff,inclusive=True,fact=4)
			gbeta=fn_rad_ker_d(beta)
			Dr=np.cos(2.*(alpha-gamma))*gbeta ; Di=np.sin(2.*(alpha-gamma))*gbeta

			dqu[0][i]=(np.dot(Dr,cq[spix]) + np.dot( Di,cu[spix]))*domega
			dqu[1][i]=(np.dot(Di,cq[spix]) + np.dot(-Dr,cu[spix]))*domega
	else:
		for i in pix_list:
			beta,alpha,gamma,spix=euler.return_euler_angles(nside,i,theta_cutoff,inclusive=True,fact=4)
			gbeta=fn_rad_ker_d(beta)
			Dr=np.cos(2.*(alpha-gamma))*gbeta ; Di=np.sin(2.*(alpha-gamma))*gbeta

			dqu[0][i]=(np.dot(Dr,q[spix]) + np.dot( Di,u[spix]))*domega
			dqu[1][i]=(np.dot(Di,q[spix]) + np.dot(-Dr,u[spix]))*domega


	return [np.zeros(npix),(q+dqu[0])*0.5,(u+dqu[1])*0.5],[np.zeros(npix),(q-dqu[0])*0.5,(u-dqu[1])*0.5]
			
# Radiation method to separate Q U into Stokes parameters corresponding to E and B modes respectively
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
			beta,alpha,gamma,spix=euler.return_euler_angles(nside,i,theta_cutoff,inclusive=True,fact=4)
			gbeta=fn_rad_ker_d(beta)
			Dr=np.cos(2.*(alpha-gamma))*gbeta ; Di=np.sin(2.*(alpha-gamma))*gbeta

			dqu[0][spix]=dqu[0][spix] + (cq[i]*Dr + cu[i]*Di)*domega
			dqu[1][spix]=dqu[1][spix] + (cq[i]*Di - cu[i]*Dr)*domega
	else:
		for i in pix_list:
			beta,alpha,gamma,spix=euler.return_euler_angles(nside,i,theta_cutoff,inclusive=True,fact=4)
			gbeta=fn_rad_ker_d(beta)
			Dr=np.cos(2.*(alpha-gamma))*gbeta ; Di=np.sin(2.*(alpha-gamma))*gbeta

			dqu[0][spix]=dqu[0][spix] + (q[i]*Dr + u[i]*Di)*domega
			dqu[1][spix]=dqu[1][spix] + (q[i]*Di - u[i]*Dr)*domega

	return [np.zeros(npix),(q+dqu[0])*0.5,(u+dqu[1])*0.5],[np.zeros(npix),(q-dqu[0])*0.5,(u-dqu[1])*0.5]
