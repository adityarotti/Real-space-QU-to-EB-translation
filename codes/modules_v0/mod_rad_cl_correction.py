import numpy as np
from scipy.special import lpmn as lpmn
from scipy.interpolate import interp1d
import numpy.polynomial.legendre as lgdr
import scipy.integrate as intg
import healpy as h
import time

def mod_gbeta(x,shift=15.,width=5.,slope=2.,expslope=2.):
	shift=shift*np.pi/180. ; width=width*np.pi/180.
	return (x**slope)*np.exp(-((x-shift)/(np.sqrt(2.)*width))**expslope)

def get_mod_gbeta_root(shift=15.,width=5.,slope=2.):
	return max(0.5*(shift + np.sqrt(shift**2. + 4.*slope*width*width)),0.5*(shift - np.sqrt(shift**2. + 4.*slope*width*width)))*np.pi/180.

def return_beta0(lmax):
	return min(180.,180.*24./lmax)*np.pi/180.

def pl2norm(l):
	return 2.*np.pi/(((l+2)*(l+1)*l*(l-1))**0.5)

def fn_step_apo(theta,beta0,frac_beta0=3.,frac_apow=0.1):
	'''
	Returns a step function with a cosine squared apodization profile
	Input variables:
	frac_beta0 = Takes the factor of beta0 as the radial cutoff.
	apow = Takes fraction of radial cutoff as in put.
	example: frac_beta0=3. implies the rcutoff=3*beta0.
		 apow=0.1. implies 10% of rcutoff is the apodization width.
	'''
	theta0=frac_beta0*beta0 ; apow=frac_apow*theta0
	xp=(theta-(theta0-apow))*np.pi/(2.*apow)
	stepfn=np.zeros(theta.size,float)
	stepfn=np.cos(xp)**2.
	stepfn[theta<=theta0-apow]=1.
	stepfn[theta>theta0]=0.
	return stepfn

def har_high_pass_filter(ell,ell0,deltaell):
    fl=np.ones(ell.size,float)
    ellp=(ell-(ell0+deltaell))*pi/(2.*deltaell)
    fl=cos(ellp)**2.
    fl[ell<ell0]=0.
    fl[ell>ell0+deltaell]=1.
    return fl

#	These function evalute the Pl0 forward and backward transforms.
def get_bl_from_beam(theta,beam,lmax):
	ell=np.arange(lmax+1)
	bl=lgdr.legfit(np.cos(theta),beam,lmax)*4.*np.pi/(2.*ell+1.)
	return bl

def get_beam_form_bl(theta,bl):
	ell=np.arange(bl.size)
	beam=lgdr.legval(np.cos(theta),((2.*ell+1.)/(4.*np.pi))*bl,tensor=False)
	return beam

class mod_rad_ker(object):
	
	def __init__(self,lmax,sampling=4000,f=3):
		
		self.lmax=lmax
		self.beta0=min(180.,180.*24./lmax)*np.pi/180. 
		self.sampling=sampling
		self.f=f ; self.f=min(self.f,np.pi/self.beta0)
		self.theta=np.linspace(0.,self.f*self.beta0,sampling)
		self.ell=np.arange(self.lmax+1)

		self.pl2=np.zeros((lmax+1,np.size(self.theta)),float)
		self.pl0=np.zeros((lmax+1,np.size(self.theta)),float)
		for i in range(self.theta.size):
		    y,temp=lpmn(2,lmax,np.cos(self.theta[i]))
		    self.pl2[:,i]=y[2,:]       
		    self.pl0[:,i]=y[0,:]       

		# Define the default radial kernel
		self.fbeta=np.zeros(self.theta.size,float)
		for i in np.arange(self.theta.size):
		    for j in np.arange(self.lmax-1):
			l=j+2
		        self.fbeta[i]=self.fbeta[i] + ((2.*l+1)/(4.*np.pi))*self.pl2[l,i]*(1./((l+2)*(l+1)*(l-1)*l)**0.5)
	
#	These functions evaluate the Pl2 forward and backward transforms.
	def get_gl_from_gbeta(self,gbeta,ulimit,lmax,rtol=1e-8,atol=1e-8,maxiter=5000):
		gl=np.zeros(lmax+1,float)
		for i in range(lmax-1):
			l=i+2
			norm=2.*np.pi/(((l+2.)*(l+1.)*l*(l-1.))**0.5)
			y=gbeta*self.pl2[l,:]*np.sin(self.theta)
			integrand=interp1d(self.theta,y,kind="cubic")
			gl[l]=intg.quadrature(integrand,0.,ulimit,rtol=rtol,tol=atol,maxiter=maxiter)[0]*norm
		return gl


	def get_gbeta_from_gl(self,gl):
		recgbeta=np.zeros(self.theta.size,float)
		lmax=gl.size-1
		for i in range(self.theta.size):
			for j in range(lmax-1):
				l=j+2
				norm=((2.*l+1)/(4.*np.pi))/(((l+2.)*(l+1.)*(l-1.)*l)**0.5)
				recgbeta[i]=recgbeta[i] + norm*gl[l]*self.pl2[l,i]
		return recgbeta






	
	
