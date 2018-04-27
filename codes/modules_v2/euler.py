import numpy as np
import healpy as h

def return_euler_angles(nside,cpix,discsize=np.pi,inclusive=False,fact=4):
	'''Returns the Euler angles for a pair of spherical coordinates'''
	theta0,phi0=h.pix2ang(nside,cpix) ; v=h.pix2vec(nside,cpix)

        spix=h.query_disc(nside,v,discsize,inclusive=inclusive,fact=fact)
        theta1,phi1=h.pix2ang(nside,spix)

	cosbeta=np.sin(theta0)*np.sin(theta1)*np.cos(phi1-phi0)+np.cos(theta0)*np.cos(theta1) 
	beta=np.arccos(cosbeta)

	na = np.sin(theta0)*np.sin(theta1)*np.sin(phi1-phi0)
	da = np.cos(theta0)*cosbeta - np.cos(theta1)
    	alpha=np.arctan2(na,da) ; alpha[abs(cosbeta)==1.]=0.

	ng = na
	dg = np.cos(theta1)*cosbeta - np.cos(theta0)
    	gamma=np.arctan2(ng,dg) ; gamma[abs(cosbeta)==1.]=0.

	return beta,alpha,gamma,spix
