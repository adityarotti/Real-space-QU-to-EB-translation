import numpy as np
import healpy as h

def return_euler_angles(theta1,phi1,theta2,phi2):
	'''Returns the Euler angles for a pair of spherical coordinates'''
	cosbeta=np.sin(theta1)*np.sin(theta2)*np.cos(phi2-phi1)+np.cos(theta1)*np.cos(theta2) 
	beta=np.arccos(cosbeta)

	na = np.sin(theta2)*np.sin(phi2-phi1)
	da = -np.sin(theta1)*np.cos(theta2) + np.sin(theta2)*np.cos(theta1)*np.cos(phi2-phi1)
    	alpha=np.arctan2(na,da)

	ng = np.sin(theta1)*np.sin(phi2-phi1)
	dg = np.sin(theta2)*np.cos(theta1) - np.sin(theta1)*np.cos(theta2)*np.cos(phi2-phi1)
    	gamma=np.arctan2(ng,dg)

	return alpha,beta,gamma


def return_spix_euler_angles(nside,cpix,discsize=np.pi,inclusive=False,fact=4):
	'''Returns the pixel numbers and the respective Euler angles within a circle of radius discsize from the central pixel: cpix'''
	theta0,phi0=h.pix2ang(nside,cpix)
	v=h.pix2vec(nside,cpix)
	spix=h.query_disc(nside,v,discsize,inclusive=inclusive,fact=fact)
	spix=np.delete(spix,np.where(spix==cpix)[0][0])
	theta1,phi1=h.pix2ang(nside,spix)

	cosbeta=np.sin(theta0)*np.sin(theta1)*np.cos(phi1-phi0)+np.cos(theta0)*np.cos(theta1) 
	beta=np.arccos(cosbeta)

	n = np.sin(theta0)*np.sin(theta1)*np.sin(phi0-phi1)

    	da = np.cos(theta0)*np.cos(beta) - np.cos(theta1)
    	alpha=np.arctan2(n,da) ; alpha[beta==0]=0. ; alpha[abs(beta-np.pi)<1e-5]=0.

	dg = np.cos(theta1)*np.cos(beta) - np.cos(theta0)
    	gamma=np.arctan2(n,dg) ; gamma[beta==0]=0. ; gamma[abs(beta-np.pi)<1e-5]=0.

	#na = np.sin(theta1)*np.sin(phi1-phi0)
	#da = -np.sin(theta0)*np.cos(theta1) + np.sin(theta1)*np.cos(theta0)*np.cos(phi1-phi0)
    	#alpha=np.arctan2(na,da)

	#ng = np.sin(theta0)*np.sin(phi1-phi0)
	#dg = np.sin(theta1)*np.cos(theta0) - np.sin(theta0)*np.cos(theta1)*np.cos(phi1-phi0)
    	#gamma=np.arctan2(ng,dg)

	return alpha,beta,gamma,spix

def fn_s2euler_alpha_gamma(nside,cpix,discsize=np.pi,inclusive=False,fact=4):
	'''Returns the pixel numbers and the respective functions of Euler angles within a circle of radius discsize from the central pixel: cpix'''
	theta0,phi0=h.pix2ang(nside,cpix)
	v=h.pix2vec(nside,cpix)
	spix=h.query_disc(nside,v,discsize,inclusive=inclusive,fact=fact)
	# Removing the central pixel from the list of surrounding pixels.
	#spix=np.delete(spix,np.where(spix==cpix)[0][0]) 
	theta1,phi1=h.pix2ang(nside,spix)
	
	cosbeta=np.sin(theta0)*np.sin(theta1)*np.cos(phi1-phi0)+np.cos(theta0)*np.cos(theta1)

	sinalpha=np.where(abs(cosbeta)<1.,-np.sin(theta1)*np.sin(phi0-phi1)/np.sqrt(1.-cosbeta*cosbeta),0.)
	cosalpha=np.where(abs(cosbeta)<1.,(np.cos(theta1) - np.cos(theta0)*cosbeta)/(np.sin(theta0)*np.sqrt(1.-cosbeta*cosbeta)),0)
	singamma=np.where(abs(cosbeta)<1.,-np.sin(theta0)*np.sin(phi0-phi1)/np.sqrt(1.-cosbeta*cosbeta),0.)
	cosgamma=np.where(abs(cosbeta)<1.,(np.cos(theta0) - np.cos(theta1)*cosbeta)/(np.sin(theta1)*np.sqrt(1.-cosbeta*cosbeta)),0)

	s2a=2.*sinalpha*cosalpha
	c2a=cosalpha*cosalpha - sinalpha*sinalpha
	s2g=2.*singamma*cosgamma
	c2g=cosgamma*cosgamma - singamma*singamma

	c2apg=c2a*c2g - s2a*s2g
	c2amg=c2a*c2g + s2a*s2g
	s2apg=s2a*c2g + c2a*s2g
	s2amg=s2a*c2g - c2a*s2g
	
	return cosbeta,c2amg,s2amg,c2apg,s2apg,spix

def fn_s2euler_alpha(nside,cpix,discsize=np.pi,inclusive=False,fact=4):
	'''Returns the pixel numbers and the respective functions of Euler angles within a circle of radius discsize from the central pixel: cpix'''
	theta0,phi0=h.pix2ang(nside,cpix)
	v=h.pix2vec(nside,cpix)
	spix=h.query_disc(nside,v,discsize,inclusive=inclusive,fact=fact)
	# Removing the central pixel from the list of surrounding pixels.
	#spix=np.delete(spix,np.where(spix==cpix)[0][0]) 
	theta1,phi1=h.pix2ang(nside,spix)
	
	cosbeta=np.sin(theta0)*np.sin(theta1)*np.cos(phi1-phi0)+np.cos(theta0)*np.cos(theta1)
	sinalpha=np.where(abs(cosbeta)<1.,-np.sin(theta1)*np.sin(phi0-phi1)/np.sqrt(1.-cosbeta*cosbeta),0.)
	cosalpha=np.where(abs(cosbeta)<1.,(np.cos(theta1) - np.cos(theta0)*cosbeta)/(np.sin(theta0)*np.sqrt(1.-cosbeta*cosbeta)),0)
	
	sin2alpha=2.*sinalpha*cosalpha
	cos2alpha=cosalpha*cosalpha - sinalpha*sinalpha
	
	return cosbeta,cos2alpha,sin2alpha,spix


def fn_s2euler_gamma(nside,cpix,discsize=np.pi,inclusive=False,fact=4):
	'''Returns the pixel numbers and the respective functions of Euler angles within a circle of radius discsize from the central pixel: cpix'''
	theta0,phi0=h.pix2ang(nside,cpix)
	v=h.pix2vec(nside,cpix)
	spix=h.query_disc(nside,v,discsize,inclusive=inclusive,fact=fact)
	# Removing the central pixel from the list of surrounding pixels.
	#spix=np.delete(spix,np.where(spix==cpix)[0][0]) 
	theta1,phi1=h.pix2ang(nside,spix)
	
	cosbeta=np.sin(theta0)*np.sin(theta1)*np.cos(phi1-phi0)+np.cos(theta0)*np.cos(theta1)
	singamma=np.where(abs(cosbeta)<1.,-np.sin(theta0)*np.sin(phi0-phi1)/np.sqrt(1.-cosbeta*cosbeta),0.)
	cosgamma=np.where(abs(cosbeta)<1.,-(np.cos(theta0) - np.cos(theta1)*cosbeta)/(np.sin(theta1)*np.sqrt(1.-cosbeta*cosbeta)),0)
	
	sin2gamma=2.*singamma*cosgamma
	cos2gamma=cosgamma*cosgamma - singamma*singamma
	
	return cosbeta,cos2gamma,sin2gamma,spix


def return_high_res_euler_angles(nside,cpix,hres_nside,discsize=np.pi,inclusive=False,fact=4):
	'''Returns the pixel numbers and the respective Euler angles within a circle of radius discsize from the central pixel: cpix'''
	theta0,phi0=h.pix2ang(nside,cpix)
	v=h.pix2vec(nside,cpix)
	spix=h.query_disc(hres_nside,v,discsize,inclusive=inclusive,fact=fact)
	theta1,phi1=h.pix2ang(hres_nside,spix)

	cosbeta=np.sin(theta0)*np.sin(theta1)*np.cos(phi1-phi0)+np.cos(theta0)*np.cos(theta1) 
	beta=np.arccos(cosbeta)

	na = np.sin(theta1)*np.sin(phi1-phi0)
	da = np.sin(theta0)*np.cos(theta1) - np.sin(theta1)*np.cos(theta0)*np.cos(phi1-phi0)
    	alpha=np.arctan2(na,da)

	ng = np.sin(theta0)*np.sin(phi1-phi0)
	dg = -np.sin(theta1)*np.cos(theta0) + np.sin(theta0)*np.cos(theta1)*np.cos(phi1-phi0)
    	gamma=np.arctan2(ng,dg)

	return alpha,beta,gamma,spix

