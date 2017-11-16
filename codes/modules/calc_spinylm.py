import numpy as np
from scipy.special import lpmn as lpmn
import healpy as h

from sympy.physics.quantum.spin import Rotation
from sympy import N

def wigd_sylm(l,m,theta,phi,s=2):
    '''Uses the Wigner D implementation in sympy.physics.quantum.spin to compute the spin spherical harmonic functions. Its very slow and is meant to be used only for testing purposes.'''
    f=Rotation.D(l,-m,s,phi,theta,0.)
    res=N(f.doit()*np.sqrt((2.*l+1.)/(4.*np.pi))*((-1.)**m))
    return res

def calc_factorial_log(x):
    '''Uses the Stirling approximation to calculate the natural log of the factorial'''
    if x==0.:
        logxf=0.
    else:
        logxf=x*np.log(x)-x + 0.5*np.log(2.*np.pi*x) + np.log(1. +  (1./(12.*x)) + (1./(288.*x*x)) - (139./(51840.*x*x*x)) - (571./(2488320.*x*x*x*x)))
    return logxf

def spin0_nlm(l,m):
    '''Returns sqrt((2l+1)/(4*pi))*sqrt((l-m)!/(l+m)!)'''
    y=0.5*(calc_factorial_log(l-m) - calc_factorial_log(l+m))
    return np.sqrt((2.*l+1.)/(4.*np.pi))*np.exp(y)

def spin2_nlm(l,m):
    '''Returns 2*sqrt((2l+1)/(4*pi))*sqrt(((l-m)!/(l+m)!)*((l-2)!/(l+2)!))'''
    y=0.5*(calc_factorial_log(l-m) - calc_factorial_log(l+m) + calc_factorial_log(l-2) - calc_factorial_log(l+2))
    return 2.*np.sqrt((2.*l+1.)/(4.*np.pi))*np.exp(y)

def return_plm(lmax,mmax,csth):	
	plm,dplm=lpmn(mmax,lmax,csth)
	return plm

def return_ylm(l,m,csth,phi,plm):
	ylm=spin0_nlm(l,m)*plm[m,l]*complex(np.cos(m*phi),np.sin(m*phi))
	return ylm

def return_p2ylm(l,m,csth,phi,plm):
	# f1lm + f2lm -- See Healpix intro to know more about these terms
	c1=-((float(l-m*m) + (m*(l-1)*csth))/(1.-csth*csth) + 0.5*l*(l-1.))
	c2=(csth + float(m))*float(l+m)/(1.-csth*csth)
	p2ylm=spin2_nlm(l,m)*(c1*plm[m,l] + c2*plm[m,l-1])#*complex(np.cos(m*phi),np.sin(m*phi))
	return p2ylm

def return_m2ylm(l,m,csth,phi,plm):
	# f1lm - f2lm -- See Healpix intro to know more about these terms
	c1=-((float(l-m*m) - (m*(l-1.)*csth))/(1.-csth*csth) + 0.5*l*(l-1.))
	c2=(csth - float(m))*float(l+m)/(1.-csth*csth)
	m2ylm=spin2_nlm(l,m)*(c1*plm[m,l] + c2*plm[m,l-1])#*complex(np.cos(m*phi),np.sin(m*phi))
	return m2ylm
