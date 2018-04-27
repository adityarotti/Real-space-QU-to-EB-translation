import numpy as np
import healpy as h

def return_euler_angles(nside,cpix,discsize=np.pi,inclusive=False,fact=4):
	'''Returns the Euler angles for a pair of spherical coordinates'''
	theta0,phi0=h.pix2ang(nside,cpix)
        v=h.pix2vec(nside,cpix)
        spix=h.query_disc(nside,v,discsize,inclusive=inclusive,fact=fact)
        theta1,phi1=h.pix2ang(nside,spix)
	cosbeta=np.sin(theta0)*np.sin(theta1)*np.cos(phi1-phi0)+np.cos(theta0)*np.cos(theta1) 
	beta=np.arccos(cosbeta)

	na = np.sin(theta0)*np.sin(theta1)*np.sin(phi1-phi0)
	da = np.cos(theta0)*cosbeta - np.cos(theta1)
    	alpha=np.arctan2(na,da)

	ng = na
	dg = np.cos(theta1)*cosbeta - np.cos(theta0)
    	gamma=np.arctan2(ng,dg)

	return beta,alpha,gamma,spix

def fn_euler_trig2alpha(nside,cpix,discsize=np.pi,inclusive=False,fact=4):
	'''Returns the pixel numbers and the respective functions of Euler angles within a circle of radius discsize from the central pixel: cpix'''
	theta0,phi0=h.pix2ang(nside,cpix)
	v=h.pix2vec(nside,cpix)
	spix=h.query_disc(nside,v,discsize,inclusive=inclusive,fact=fact)
	# Removing the central pixel from the list of surrounding pixels.
	#spix=np.delete(spix,np.where(spix==cpix)[0][0]) 
	theta1,phi1=h.pix2ang(nside,spix)
	
	cosbeta=np.sin(theta0)*np.sin(theta1)*np.cos(phi1-phi0)+np.cos(theta0)*np.cos(theta1)
	sinalpha=np.where(abs(cosbeta)<1.,np.sin(theta1)*np.sin(phi1-phi0)/np.sqrt(1.-cosbeta*cosbeta),0.)
	cosalpha=np.where(abs(cosbeta)<1.,(np.cos(theta0)*cosbeta - np.cos(theta1))/(np.sin(theta0)*np.sqrt(1.-cosbeta*cosbeta)),1.)
	
	sin2alpha=2.*sinalpha*cosalpha
	cos2alpha=cosalpha*cosalpha - sinalpha*sinalpha
	
	return cosbeta,cos2alpha,sin2alpha,spix

def fn_euler_trig2gamma(nside,cpix,discsize=np.pi,inclusive=False,fact=4):
	'''Returns the pixel numbers and the respective functions of Euler angles within a circle of radius discsize from the central pixel: cpix'''
	theta0,phi0=h.pix2ang(nside,cpix)
	v=h.pix2vec(nside,cpix)
	spix=h.query_disc(nside,v,discsize,inclusive=inclusive,fact=fact)
	# Removing the central pixel from the list of surrounding pixels.
	#spix=np.delete(spix,np.where(spix==cpix)[0][0]) 
	theta1,phi1=h.pix2ang(nside,spix)
	
	cosbeta=np.sin(theta0)*np.sin(theta1)*np.cos(phi1-phi0)+np.cos(theta0)*np.cos(theta1)
	singamma=np.where(abs(cosbeta)<1.,-np.sin(theta0)*np.sin(phi1-phi0)/np.sqrt(1.-cosbeta*cosbeta),0.)
	cosgamma=np.where(abs(cosbeta)<1.,(np.cos(theta0) - cosbeta*np.cos(theta1))/(np.sin(theta1)*np.sqrt(1.-cosbeta*cosbeta)),1.)
	
	sin2gamma=2.*singamma*cosgamma
	cos2gamma=cosgamma*cosgamma - singamma*singamma
	
	return cosbeta,cos2gamma,sin2gamma,spix

def fn_euler_trig2_alpha_plus_gamma(nside,cpix,discsize=np.pi,inclusive=False,fact=4):
	'''Returns the pixel numbers and the respective functions of Euler angles within a circle of radius discsize from the central pixel: cpix'''
	theta0,phi0=h.pix2ang(nside,cpix)
	v=h.pix2vec(nside,cpix)
	spix=h.query_disc(nside,v,discsize,inclusive=inclusive,fact=fact)
	# Removing the central pixel from the list of surrounding pixels.
	#spix=np.delete(spix,np.where(spix==cpix)[0][0]) 
	theta1,phi1=h.pix2ang(nside,spix)
	
	cosbeta=np.sin(theta0)*np.sin(theta1)*np.cos(phi1-phi0)+np.cos(theta0)*np.cos(theta1)
	sinalpha=np.where(abs(cosbeta)<1.,np.sin(theta1)*np.sin(phi1-phi0)/np.sqrt(1.-cosbeta*cosbeta),0.)
	cosalpha=np.where(abs(cosbeta)<1.,(np.cos(theta0)*cosbeta - np.cos(theta1))/(np.sin(theta0)*np.sqrt(1.-cosbeta*cosbeta)),1.)
	singamma=np.where(abs(cosbeta)<1.,-np.sin(theta0)*np.sin(phi1-phi0)/np.sqrt(1.-cosbeta*cosbeta),0.)
	cosgamma=np.where(abs(cosbeta)<1.,(np.cos(theta0) - cosbeta*np.cos(theta1))/(np.sin(theta1)*np.sqrt(1.-cosbeta*cosbeta)),1.)

	s2a=2.*sinalpha*cosalpha ; c2a=cosalpha*cosalpha - sinalpha*sinalpha
	s2g=2.*singamma*cosgamma ; c2g=cosgamma*cosgamma - singamma*singamma

	c2apg=c2a*c2g - s2a*s2g
	s2apg=s2a*c2g + c2a*s2g
	
	return cosbeta,c2apg,s2apg,spix

def fn_euler_trig2_alpha_minus_gamma(nside,cpix,discsize=np.pi,inclusive=False,fact=4):
	'''Returns the pixel numbers and the respective functions of Euler angles within a circle of radius discsize from the central pixel: cpix'''
	theta0,phi0=h.pix2ang(nside,cpix)
	v=h.pix2vec(nside,cpix)
	spix=h.query_disc(nside,v,discsize,inclusive=inclusive,fact=fact)
	# Removing the central pixel from the list of surrounding pixels.
	#spix=np.delete(spix,np.where(spix==cpix)[0][0]) 
	theta1,phi1=h.pix2ang(nside,spix)
	
	cosbeta=np.sin(theta0)*np.sin(theta1)*np.cos(phi1-phi0)+np.cos(theta0)*np.cos(theta1)
	sinalpha=np.where(abs(cosbeta)<1.,np.sin(theta1)*np.sin(phi1-phi0)/np.sqrt(1.-cosbeta*cosbeta),0.)
	cosalpha=np.where(abs(cosbeta)<1.,(np.cos(theta0)*cosbeta - np.cos(theta1))/(np.sin(theta0)*np.sqrt(1.-cosbeta*cosbeta)),1.)
	singamma=np.where(abs(cosbeta)<1.,-np.sin(theta0)*np.sin(phi1-phi0)/np.sqrt(1.-cosbeta*cosbeta),0.)
	cosgamma=np.where(abs(cosbeta)<1.,(np.cos(theta0) - cosbeta*np.cos(theta1))/(np.sin(theta1)*np.sqrt(1.-cosbeta*cosbeta)),1.)

	s2a=2.*sinalpha*cosalpha ; c2a=cosalpha*cosalpha - sinalpha*sinalpha
	s2g=2.*singamma*cosgamma ; c2g=cosgamma*cosgamma - singamma*singamma

	c2amg=c2a*c2g + s2a*s2g
	s2amg=s2a*c2g - c2a*s2g
	
	return cosbeta,c2amg,s2amg,spix
