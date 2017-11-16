from scipy.special import lpmn as lpmn
import numpy.polynomial.legendre as lgdr
from scipy.interpolate import interp1d
import scipy.integrate as intg
import numpy as np
import sys

def mod_rad_ker(theta,shift=0.2,width=0.5,amp=1.,slope=2.,expslope=2.):
	rad_ker=((amp*theta)**slope)*np.exp(-((theta-shift)/(np.sqrt(2.)*width))**expslope)
	return rad_ker

def fn_apodization(theta,theta_cutoff=np.pi/6.,apow_frac=0.2):
	fnapo=np.zeros(theta.size,"double")
	apow=apow_frac*theta_cutoff
	# Cosine squared apodization.
	x=(theta-(theta_cutoff-apow))*np.pi/(2.*apow)
	fnapo=np.cos(x)**2.
	fnapo[theta<=theta_cutoff-apow]=1.
	fnapo[theta>theta_cutoff]=0.
	return fnapo

def get_plm0(lmax,sampling,theta_min=0.,theta_max=np.pi,m0=2,theta=np.zeros(0,"double")):
	'''This is written to get the associated Legendre polynomials at a fixed m0. The default m0=2
	When the array costheta is provided, the sampling parameter is ignored.
	'''
	if np.size(theta)==0:
		theta=np.linspace(theta_min,theta_max,sampling)

	csth=np.cos(theta)
	pl2=np.zeros((lmax+1,np.size(csth)),"double")
	for i in range(np.size(csth)):
		plm,dplm=lpmn(m0,lmax,csth[i])
		for l in range(lmax+1):
			pl2[l,i]=plm[m0,l]
	return theta,pl2

# This is the default radial kernel.
def calc_qu2eb_rad_ker(lmax,theta,pl2,lmin=2):
	if lmax>np.size(pl2[:,0])-1:
		#print "lmax given:",lmax
		#print "Maximum multipole of Legendre polynomials:", np.size(pl2[:,0])-1
		sys.exit("Exit error: lmax greater than that computed for the Legendre polynomials.\n Maximum lmax available:" + str(np.size(pl2[:,0])-1))
	else:
		qu2eb_rad_ker=np.zeros(np.size(theta),"double")
		for i in range(np.size(theta)):
			for j in range(lmax-lmin+1):
				l=lmin+j
				qu2eb_rad_ker[i]=qu2eb_rad_ker[i] + ((2.*l+1.)/(4.*np.pi))*pl2[l,i]/(((l+2.)*(l+1.)*(l-1.)*l)**0.5)
	return qu2eb_rad_ker
		
	
def calc_qu2queb_rad_ker(lmax,theta,pl2,lmin=2,theta_eps=1e-6):
	if lmax>np.size(pl2[:,0])-1:
		#print "lmax given:",lmax
		#print "Maximum multipole of Legendre polynomials:", np.size(pl2[:,0])-1
		sys.exit("Exit error: lmax greater than that computed for the Legendre polynomials.\n Maximum lmax available:" + str(np.size(pl2[:,0])-1))
	else:
		csth=np.cos(theta)
		snth=np.sin(theta)
		rad_ker_p2=np.zeros(np.size(theta),"double")
		rad_ker_m2=np.zeros(np.size(theta),"double")

		for j in range(lmax-lmin+1):
			l=lmin+j
			for i in range(np.size(theta)):
				if theta[i] >  theta_eps and abs(theta[i]-np.pi) > theta_eps:
					c1=(((l-4.) + (2.*(l-1.)*csth[i]))/(snth[i]**2.) + 0.5*l*(l-1.))
					c2=(csth[i] + 2.)*(l+2.)/(snth[i]**2.)
					rad_ker_p2[i]=rad_ker_p2[i] + 2.*(-c1*pl2[l,i] + c2*pl2[l-1,i])*((2.*l+1.)/(4.*np.pi))/((l+2.)*(l+1.)*(l-1.)*l)

					c1=(((l-4.) - (2.*(l-1.)*csth[i]))/(snth[i]**2.) + 0.5*l*(l-1.))
					c2=(csth[i] - 2.)*(l+2.)/(snth[i]**2.)
					rad_ker_m2[i]=rad_ker_m2[i] + 2.*(-c1*pl2[l,i] + c2*pl2[l-1,i])*((2.*l+1.)/(4.*np.pi))/((l+2.)*(l+1.)*(l-1.)*l)

				elif theta[i] < theta_eps:
					c1=(((l-4.) + (2.*(l-1.)*csth[i])) + 0.5*l*(l-1.)*(snth[i]**2.))
					c2=(csth[i] + 2.)*(l-2.)
					rad_ker_p2[i]=rad_ker_p2[i] + 0.25*(-c1 + c2)*((2.*l+1.)/(4.*np.pi))
	
					c1=(((l-4.) - (2.*(l-1.)*csth[i])) + 0.5*l*(l-1.)*(snth[i]**2.))
					c2=(csth[i] - 2.)*(l-2.)
					rad_ker_m2[i]=rad_ker_m2[i] + 0.25*(-c1 + c2)*((2.*l+1.)/(4.*np.pi))

				elif abs(theta[i]-np.pi) < theta_eps:
					c1=(((l-4.) + (2.*(l-1.)*csth[i])) + 0.5*l*(l-1.)*(snth[i]**2.))*((-1.)**float(l))
					c2=((csth[i] + 2.)*(l-2.))*((-1.)**float(l-1))
					rad_ker_p2[i]=rad_ker_p2[i] + 0.25*(-c1 + c2)*((2.*l+1.)/(4.*np.pi))
	
					c1=(((l-4.) - (2.*(l-1.)*csth[i])) + 0.5*l*(l-1.)*(snth[i]**2.))*((-1.)**float(l))
					c2=((csth[i] - 2.)*(l-2.))*((-1.)**float(l-1))
					rad_ker_m2[i]=rad_ker_m2[i] + 0.25*(-c1 + c2)*((2.*l+1.)/(4.*np.pi))
	return rad_ker_p2,rad_ker_m2


def get_gl_from_gbeta(theta,rad_ker,pl2,theta_cutoff,lmax,rtol=1e-8,atol=1e-8,maxiter=5000):
	gl=np.zeros(lmax+1,"double")
	for i in range(lmax-1):
		l=i+2
		norm=2.*np.pi/(((l+2.)*(l+1.)*l*(l-1.))**0.5)
		y=rad_ker*pl2[l,]*np.sin(theta)
		integrand=interp1d(theta,y,kind="cubic")
		gl[l]=intg.quadrature(integrand,0.,theta_cutoff,rtol=rtol,tol=atol,maxiter=maxiter)[0]*norm
	return gl


def get_gbeta_from_gl(theta,gl,pl2):
	rad_ker=np.zeros(np.size(theta),"double")
	lmax=np.size(gl)-1
	for i in range(np.size(theta)):
		for j in range(lmax-1):
			l=j+2
			norm=((2.*l+1)/(4.*np.pi))/(((l+2.)*(l+1.)*(l-1.)*l)**0.5)
			rad_ker[i]=rad_ker[i] + norm*gl[l]*pl2[l,i]
	return rad_ker


def get_bl_from_beam(theta,beam,lmax):
	ell=np.arange(lmax+1)
	bl=lgdr.legfit(np.cos(theta),beam,lmax)*(4.*np.pi/(2.*ell+1.))
	return bl


def get_beam_form_bl(theta,bl):
	ell=np.arange(bl.size)
	beam=lgdr.legval(np.cos(theta),((2.*ell+1.)/(4.*np.pi))*bl,tensor=False)
	return beam

