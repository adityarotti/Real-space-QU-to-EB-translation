import numpy as np
from scipy.special import lpmn as lpmn
from scipy.interpolate import interp1d
import healpy as h

class real_space_queb_kernels(object):

	def __init__(self,nside,tmin=0.,tmax=30.,maxlmax=1024,sampling=1024):
		self.nside=nside
		self.tmin=tmin
		self.tmax=tmax
		self.maxlmax=maxlmax
		self.sampling=sampling		

		self.npix=h.nside2npix(self.nside)
		self.omega=4.*np.pi/self.npix

		self.theta=np.linspace(tmin*np.pi/180.,tmax*np.pi/180.,self.sampling)
		self.pl2=np.zeros((self.maxlmax+1,np.size(self.theta)),float)
		self.taper=np.ones(self.theta.size,float)


		for i in range(self.theta.size):
			y,temp=lpmn(2,self.maxlmax,np.cos(self.theta[i]))
			for l in range(self.maxlmax+1):
				self.pl2[l,i]=y[2,l]

#	-----------------------------------------------------------------------------------------------------------------------------------------
#	Function called by other internal functions.
	def gauss(self,x,sigma=180.):
	    y=np.exp(-x**2./(2.*sigma*sigma))#/sqrt(2.*pi*sigma*sigma)
	    return y

	def return_euler_angles(self,nside,cpixel,discsize=180.,nest=False):
		theta1,phi1=h.pix2ang(nside,cpixel,nest=nest)

		v=h.pix2vec(nside,cpixel,nest=nest)
		spixel=h.query_disc(nside,v,discsize*np.pi/180.,inclusive=True,fact=4,nest=nest)
		theta2,phi2=h.pix2ang(nside,spixel,nest=nest)

		temp_beta=np.sin(theta1)*np.sin(theta2)*np.cos(phi2-phi1)+np.cos(theta1)*np.cos(theta2) 
		temp_beta[temp_beta>1.]=1. ; temp_beta[temp_beta<-1.]=-1.
		beta=np.arccos(temp_beta)

		n = np.sin(theta1)*np.sin(theta2)*np.sin(phi1-phi2)

    		da = np.cos(theta1)*np.cos(beta) - np.cos(theta2)
    		alpha=np.arctan2(n,da) ; alpha[beta==0]=0. ; alpha[abs(beta-np.pi)<1e-5]=0.

		dg = np.cos(theta2)*np.cos(beta) - np.cos(theta1)
    		gamma=np.arctan2(n,dg) ; gamma[beta==0]=0. ; gamma[abs(beta-np.pi)<1e-5]=0.

		return alpha,beta,gamma,spixel
#	-----------------------------------------------------------------------------------------------------------------------------------------
	
#	-----------------------------------------------------------------------------------------------------------------------------------------
#	Function to evaluate the radial kernels.
	def calc_qu2eb_rad_kernel(self,lmax,lmin=2):
		self.rad_ker=np.zeros(np.size(self.theta),float)
		if lmax <= self.maxlmax:				
			self.rad_ker[:]=0.
			for i in range(self.theta.size):
				for j in range(lmax-lmin+1):
					l=j+lmin
					norm=self.omega*((2.*l+1.)/(4.*np.pi))*((1./((l+2.)*(l+1.)*(l-1.)*l))**0.5)
					self.rad_ker[i]=self.rad_ker[i] + norm*self.pl2[l,i]
		else:
			print "Please provide an lmax less than maxlmax:" + str(self.maxlmax) + "or increase the maxlmax"
	
	def calc_qu2queb_rad_kernel(self,lmax,lmin=2):
		thetamin=np.sqrt(self.omega)/10.
		self.rad_ker_p2=np.zeros(np.size(self.theta),float)
		self.rad_ker_m2=np.zeros(np.size(self.theta),float)
		if lmax <= self.maxlmax:
			self.rad_ker_p2[:]=0.
			self.rad_ker_m2[:]=0.

			for i in range(self.theta.size):
				for j in range(lmax-lmin+1):
					l=j+lmin

					if self.theta[i]>=thetamin and self.theta[i]<=np.pi-thetamin:
						norm=self.omega*((2.*l+1.)/(4.*np.pi))*(2./((l+2.)*(l+1.)*(l-1.)*l))

						c1=(l-4.)/(np.sin(self.theta[i])**2.) + 0.5*l*(l-1.) + 2.*(l-1.)*np.cos(self.theta[i])/(np.sin(self.theta[i])**2.)
				                c2=((l+2.)*np.cos(self.theta[i]) + 2.*(l+2.))/(np.sin(self.theta[i])**2.)		
						self.rad_ker_p2[i]=self.rad_ker_p2[i] + norm*(-c1*self.pl2[l,i] + c2*self.pl2[l-1,i])

						c1=(l-4.)/(np.sin(self.theta[i])**2.) + 0.5*l*(l-1.) - 2.*(l-1.)*np.cos(self.theta[i])/(np.sin(self.theta[i])**2.)
				                c2=((l+2.)*np.cos(self.theta[i]) - 2.*(l+2.))/(np.sin(self.theta[i])**2.)		
						self.rad_ker_m2[i]=self.rad_ker_m2[i] + norm*(-c1*self.pl2[l,i] + c2*self.pl2[l-1,i])
					elif self.theta[i]<thetamin:
						norm=self.omega*((2.*l+1)/(4.*np.pi))*(1./4.)
						
						c1=(l-4.) + 0.5*l*(l-1.)*(np.sin(self.theta[i])**2.) + 2.*(l-1.)*np.cos(self.theta[i])
				                c2=((l-2.)*np.cos(self.theta[i]) + 2.*(l-2.))
						self.rad_ker_p2[i]=self.rad_ker_p2[i] + norm*(-c1 + c2)

						c1=(l-4.) + 0.5*l*(l-1.)*(np.sin(self.theta[i])**2.) - 2.*(l-1.)*np.cos(self.theta[i])
				                c2=((l-2.)*np.cos(self.theta[i]) - 2.*(l-2.))
						self.rad_ker_m2[i]=self.rad_ker_m2[i] + norm*(-c1 + c2)
					elif self.theta[i] > np.pi-thetamin:
						norm=self.omega*((2.*l+1)/(4.*np.pi))*(1./4.)
						
						c1=((l-4.) + 0.5*l*(l-1.)*(np.sin(self.theta[i])**2.) + 2.*(l-1.)*np.cos(self.theta[i]))*((-1.)**float(l))
				                c2=((l-2.)*np.cos(self.theta[i]) + 2.*(l-2.))*((-1.)**float(l-1))
						self.rad_ker_p2[i]=self.rad_ker_p2[i] + norm*(-c1 + c2)

						c1=((l-4.) + 0.5*l*(l-1.)*(np.sin(self.theta[i])**2.) - 2.*(l-1.)*np.cos(self.theta[i]))*((-1.)**float(l))
				                c2=((l-2.)*np.cos(self.theta[i]) - 2.*(l-2.))*((-1.)**float(l-1))
						self.rad_ker_m2[i]=self.rad_ker_m2[i] + norm*(-c1 + c2)
		else:
			print "Please provide an lmax less than maxlmax:" + str(self.maxlmax) + "or increase the maxlmax"

	# Zaldariagga radial kernel.
	def setup_qu2eb_thetasqinv(self,norm=1.):
		self.rad_thetasqinv=np.zeros(np.size(self.theta),float)
		for i in range(self.theta.size):
			if self.theta[i]>0.:
				self.rad_thetasqinv[i]=norm/((self.theta[i])**2.)
			else:
				self.rad_thetasqinv[i]=0.
		self.fn_rad_thetaqsinv=interp1d(self.theta,self.rad_thetasqinv,assume_sorted=True,kind="cubic",bounds_error=False,fill_value=0.0)


	def setup_fn_rad_ker(self):
		self.fn_rad_ker=interp1d(self.theta,self.rad_ker*self.taper,assume_sorted=True,kind="cubic",bounds_error=False,fill_value=0.0)		
		self.fn_rad_ker_p2=interp1d(self.theta,self.rad_ker_p2*self.taper,assume_sorted=True,kind="cubic",bounds_error=False,fill_value=0.0)
		self.fn_rad_ker_m2=interp1d(self.theta,self.rad_ker_m2*self.taper,assume_sorted=True,kind="cubic",bounds_error=False,fill_value=0.0)

	def setup_taper(self,theta_max_cutoff=180., apow=2.,sigma_cutoff=4.,tapertype=1):
		'''
		tapertype=1 does a cosine squared apodization profile.\n
		tapertype=2 does a gassuian apodization profile.
		'''
		self.taper=np.ones(self.theta.size,float)
		theta_max_cutoff=theta_max_cutoff*np.pi/180. ; apow=apow*np.pi/180.

		if theta_max_cutoff>=self.tmax:
			self.taper[:]=1.
		else:
			if tapertype==1:
				self.taper=((self.theta-(theta_max_cutoff-apow))/(2.*apow))*np.pi
				self.taper=np.cos(self.taper)**2.
				self.taper[self.theta<theta_max_cutoff-apow]=1.
				self.taper[self.theta>theta_max_cutoff]=0.
			elif tapertype==2:
				self.taper=self.gauss((self.theta-(theta_max_cutoff-sigma_cutoff*apow)),apow)
				self.taper[self.theta<theta_max_cutoff-sigma_cutoff*apow]=1.
			else:
				self.taper=((self.theta-(theta_max_cutoff-apow))/apow)*np.pi
				self.taper=(1.+np.cos(self.taper))/2.
				self.taper[self.theta<theta_max_cutoff-apow]=1.
				self.taper[self.theta>theta_max_cutoff]=0.
		self.fn_taper=interp1d(self.theta,self.taper,assume_sorted=True,kind="cubic",bounds_error=False,fill_value=0.0)		
#	-----------------------------------------------------------------------------------------------------------------------------------------


#	-----------------------------------------------------------------------------------------------------------------------------------------
#	Convolution kernels.

	def convert_qu2eb(self,tqu,discsize=180.,pindex=[],nest=False):
		
		if h.get_nside(tqu) != self.nside:
			print "Map not at expected resolution of nside:" + str(self.nside)
			print "Upgrading the map to the required resolution before evaluation."
			tqu=h.ud_grade(tqu,self.nside)
	
		if nest:
			tqu=h.reorder(tqu,r2n=True)


		# Setting the pixel index
		if np.size(pindex)==0:
			print "Computing on all pixels"
			pindex=np.arange(self.npix)

		eb=[np.zeros(self.npix,float),np.zeros(self.npix,float)]
		for i in pindex:
			alpha,beta,gamma,pix2=self.return_euler_angles(self.nside,i,discsize=discsize,nest=nest)
			eb[0][i]=np.dot(-np.cos(2.*alpha)*self.fn_rad_ker(beta),tqu[1][pix2]) + np.dot(np.sin(2.*alpha)*self.fn_rad_ker(beta),tqu[2][pix2])
			eb[1][i]=np.dot(-np.sin(2.*alpha)*self.fn_rad_ker(beta),tqu[1][pix2]) + np.dot(-np.cos(2.*alpha)*self.fn_rad_ker(beta),tqu[2][pix2])
			
		if nest:
			eb=h.reorder(eb,n2r=True)

		return [np.zeros(self.npix),eb[0],eb[1]]
			
	
	def convert_eb2qu(self,teb,discsize=180.,pindex=[],nest=False):
		
		if h.get_nside(teb) != self.nside:
			print "Map not at expected resolution of nside:" + str(self.nside)
			print "Upgrading the map to the required resolution before evaluation."
			teb=h.ud_grade(teb,self.nside)

		if nest:
			teb=h.reorder(teb,r2n=True)

		# Setting the pixel index
		if np.size(pindex)==0:
			pindex=np.arange(self.npix)

		qu=[np.zeros(self.npix,float),np.zeros(self.npix,float)]
		for i in pindex:
			alpha,beta,gamma,pix2=self.return_euler_angles(self.nside,i,discsize=discsize,nest=nest)
			qu[0][i]=np.dot(-np.cos(2.*alpha)*self.fn_rad_ker(beta),teb[1][pix2]) + np.dot(-np.sin(2.*alpha)*self.fn_rad_ker(beta),teb[2][pix2])
			qu[1][i]=np.dot(np.sin(2.*alpha)*self.fn_rad_ker(beta),teb[1][pix2]) + np.dot(-np.cos(2.*alpha)*self.fn_rad_ker(beta),teb[2][pix2])
			
		if nest:
			qu=h.reorder(qu,n2r=True)

		return [np.zeros(self.npix),qu[0],qu[1]]
		
		
	def decompose_qu2_equ_bqu(self,tqu,discsize=180.,pindex=[],nest=False):

		if h.get_nside(tqu) != self.nside:
			print "Map not at expected resolution of nside:" + str(self.nside)
			print "Upgrading the map to the required resolution before evaluation."
			tqu=h.ud_grade(tqu,self.nside)
	
		if nest:
			tqu=h.reorder(tqu,r2n=True)

		# Setting the pixel index
		if np.size(pindex)==0:
			pindex=np.arange(self.npix)

		equ=[np.zeros(self.npix,float),np.zeros(self.npix,float)]
		bqu=[np.zeros(self.npix,float),np.zeros(self.npix,float)]
		for i in pindex:
			alpha,beta,gamma,pix2=self.return_euler_angles(self.nside,i,discsize=discsize,nest=nest)

			Ir=0.5*np.cos(2.*(alpha+gamma))*self.fn_rad_ker_m2(beta)
			Ii=0.5*np.sin(2.*(alpha+gamma))*self.fn_rad_ker_m2(beta)
			Mr=0.5*np.cos(2.*(alpha-gamma))*self.fn_rad_ker_p2(beta)
			Mi=0.5*np.sin(2.*(alpha-gamma))*self.fn_rad_ker_p2(beta)

			equ[0][i]=np.dot(Ir+Mr,tqu[1][pix2]) + np.dot(Ii-Mi,tqu[2][pix2])
			equ[1][i]=np.dot(-Ii-Mi,tqu[1][pix2]) + np.dot(Ir-Mr,tqu[2][pix2])

			bqu[0][i]=np.dot(Ir-Mr,tqu[1][pix2]) + np.dot(Ii+Mi,tqu[2][pix2])
			bqu[1][i]=np.dot(-Ii+Mi,tqu[1][pix2]) + np.dot(Ir+Mr,tqu[2][pix2])

		if nest:
			equ=h.reorder(equ,n2r=True)
			bqu=h.reorder(bqu,n2r=True)

		return [np.zeros(self.npix,float),equ[0],equ[1]],[np.zeros(self.npix,float),bqu[0],bqu[1]]


	def convert_qu2eb_box(self,tqu,discsize=180.,pindex=[],nest=False):
		
		if h.get_nside(tqu) != self.nside:
			print "Map not at expected resolution of nside:" + str(self.nside)
			print "Upgrading the map to the required resolution before evaluation."
			tqu=h.ud_grade(tqu,self.nside)

		if nest:
			tqu=h.reorder(tqu,r2n=True)

		# Setting the pixel index
		if np.size(pindex)==0:
			pindex=np.arange(self.npix)

		eb=[np.zeros(self.npix,float),np.zeros(self.npix,float)]
		for i in pindex:
			alpha,beta,gamma,pix2=self.return_euler_angles(self.nside,i,discsize=discsize,nest=nest)
			eb[0][i]=np.dot(-np.cos(2.*alpha),tqu[1][pix2]) + np.dot( np.sin(2.*alpha),tqu[2][pix2])
			eb[1][i]=np.dot(-np.sin(2.*alpha),tqu[1][pix2]) + np.dot(-np.cos(2.*alpha),tqu[2][pix2])

		if nest:
			eb=h.reorder(eb,n2r=True)
			
		return [np.zeros(self.npix),eb[0],eb[1]]


	def convert_qu2eb_thetasqinv(self,tqu,discsize=180.,pindex=[],nest=False):
		
		if h.get_nside(tqu) != self.nside:
			print "Map not at expected resolution of nside:" + str(self.nside)
			print "Upgrading the map to the required resolution before evaluation."
			tqu=h.ud_grade(tqu,self.nside)

		if nest:
			tqu=h.reorder(tqu,r2n=True)

		# Setting the pixel index
		if np.size(pindex)==0:
			pindex=np.arange(self.npix)

		eb=[np.zeros(self.npix,float),np.zeros(self.npix,float)]
		for i in pindex:
			alpha,beta,gamma,pix2=self.return_euler_angles(self.nside,i,discsize=discsize,nest=nest)
			eb[0][i]=np.dot(-np.cos(2.*alpha)*self.fn_rad_thetaqsinv(beta),tqu[1][pix2]) + np.dot( np.sin(2.*alpha)*self.fn_rad_thetaqsinv(beta),tqu[2][pix2])
			eb[1][i]=np.dot(-np.sin(2.*alpha)*self.fn_rad_thetaqsinv(beta),tqu[1][pix2]) + np.dot(-np.cos(2.*alpha)*self.fn_rad_thetaqsinv(beta),tqu[2][pix2])

		if nest:
			eb=h.reorder(eb,n2r=True)
			
		return [np.zeros(self.npix),eb[0],eb[1]]		
	
	def convert_eb2qu_box(self,teb,discsize=180.,pindex=[],nest=False):
		
		if h.get_nside(teb) != self.nside:
			print "Map not at expected resolution of nside:" + str(self.nside)
			print "Upgrading the map to the required resolution before evaluation."
			teb=h.ud_grade(teb,self.nside)

		if nest:
			teb=h.reorder(teb,r2n=True)


		# Setting the pixel index
		if np.size(pindex)==0:
			pindex=np.arange(self.npix)

		qu=[np.zeros(self.npix,float),np.zeros(self.npix,float)]
		for i in pindex:
			alpha,beta,gamma,pix2=self.return_euler_angles(self.nside,i,discsize=discsize,nest=nest)
			qu[0][i]=np.dot(-np.cos(2.*alpha),teb[1][pix2]) + np.dot(-np.sin(2.*alpha),teb[2][pix2])
			qu[1][i]=np.dot( np.sin(2.*alpha),teb[1][pix2]) + np.dot(-np.cos(2.*alpha),teb[2][pix2])

		if nest:
			qu=h.reorder(qu,n2r=True)
			
		return [np.zeros(self.npix),qu[0],qu[1]]	
#	-----------------------------------------------------------------------------------------------------------------------------------------

#	-----------------------------------------------------------------------------------------------------------------------------------------
# 	Returns kernels for plotting.
	def return_qu2eb_kernel(self,pix,discsize=180.,nest=False):
		'''
		Returns the kernel which translated Q,U to E,B around the specified pixel: pix
		The kernel is only evaluated in the pixels which lie within the specified discsize: discsize in degrees.
		'''
		alpha,beta,gamma=self.return_euler_angles(pix,discsize=discsize,nest=nest)

		k11=-np.cos(2.*alpha)*self.fn_rad_ker(beta) 
		k12=-np.sin(2.*alpha)*self.fn_rad_ker(beta)
		k21= np.sin(2.*alpha)*self.fn_rad_ker(beta)
		k22=-np.cos(2.*alpha)*self.fn_rad_ker(beta)
		
		return [(k11,k12),(k21,k22)]


	def return_qu2_equ_bqu_kernel(self,pix,discsize=180.,nest=False):
		'''
		Returns the kernel which translated Q,U to Q,U corresponding to E & B around the specified pixel: pix
		The kernel is only evaluated in the pixels which lie within the specified discsize: discsize in degrees.
		'''
		alpha,beta,gamma=self.return_euler_angles(self.nside,pix,discsize=discsize,nest=nest)

		Ir=0.5*np.cos(2.*(alpha+gamma))*self.fn_rad_ker_m2(beta)
		Ii=0.5*np.sin(2.*(alpha+gamma))*self.fn_rad_ker_m2(beta)
		Mr=0.5*np.cos(2.*(alpha-gamma))*self.fn_rad_ker_p2(beta)
		Mi=0.5*np.sin(2.*(alpha-gamma))*self.fn_rad_ker_p2(beta)

		return [(Ir+Mr,Ii-Mi),(-Ii-Mi,Ir-Mr)],[(Ir-Mr,Ii+Mi),(-Ii+Mi,Ir+Mr)]
		#return [(Ir,Ii),(Mr,Mi)]
#	-----------------------------------------------------------------------------------------------------------------------------------------


#	-----------------------------------------------------------------------------------------------------------------------------------------
#	Returns the mask pixels which have pixels where E and B are reliably estimated for a simple band mask.
	def return_valib_ebpixels_from_mask(self,mask,discsize=180.,nest=False):

		if h.get_nside(mask) != self.nside:
			print "Map not at expected resolution of nside:" + str(self.nside)
			print "Upgrading the map to the required resolution before evaluation."
			mask=h.ud_grade(mask,self.nside)
		if nest:
			mask=h.reorder(mask,r2n=True)

		v=h.pix2vec(self.nside,0,nest=nest)
		spixel=h.query_disc(self.nside,v,discsize*np.pi/180.,inclusive=True,fact=4,nest=nest)
		norm=float(np.size(spixel))

		validpix=np.zeros(self.npix,float)
		for i in range(self.npix):
			alpha,beta,gamma,pix2=self.return_euler_angles(i,discsize=discsize,nest=nest)
			validpix[i]=np.sum(mask[pix2])/norm

		if nest:
			validpix=h.reorder(validpix,n2r=True)

		return validpix
#	-----------------------------------------------------------------------------------------------------------------------------------------
