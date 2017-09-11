import numpy as np

def return_euler(theta1,phi1,theta2,phi2):

	n = np.sin(theta1)*np.sin(theta2)*np.cos(phi1-phi2) + np.cos(theta1)*np.cos(theta2)
	beta = np.arccos(n)
	
	n = np.sin(theta1)*np.sin(theta2)*np.sin(phi1-phi2)
	da = np.cos(theta1)*np.cos(beta) - np.cos(theta2)
	dg = np.cos(theta2)*np.cos(beta) - np.cos(theta1)


        if n>=0 and da>=0 :
		alpha=np.arctan(n/da)
	elif n>=0 and da<0:
        	alpha=np.pi-np.arctan(n/abs(da))
	elif n<0 and da<0:
        	alpha=np.arctan(abs(n)/abs(da))-np.pi
        elif n<0 and da>0:
                alpha=2.*np.pi-np.arctan(abs(n)/da)

	if n==0:
		alpha=0.0
	alpha=np.arctan2(n,da)


        if n>=0 and dg>=0 :
		gamma=np.arctan(n/dg)
	elif n>=0 and dg<0:
        	gamma=np.pi-np.arctan(n/abs(dg))
	elif n<0 and dg<0:
        	gamma=np.pi+np.arctan(abs(n)/abs(dg))
        elif n<0 and dg>0:
                gamma=2.*np.pi-np.arctan(abs(n)/dg)	

	if n==0:
		gamma=0.0

	#gamma=2.*np.pi-gamma
	gamma=np.arctan2(n,dg)

	return alpha,beta,gamma,n,da,dg	

def return_euler_tan_old(theta1,phi1,theta2,phi2):

	n = np.sin(theta1)*np.sin(theta2)*np.cos(phi1-phi2) + np.cos(theta1)*np.cos(theta2)
	beta = np.arccos(n)
	
	n = np.sin(theta2)*np.sin(phi1-phi2)
	d = np.sin(theta2)*np.cos(theta1)*np.cos(phi1-phi2) - np.sin(theta1)*np.cos(theta2)

        if n>=0 and d>=0 :
		alpha=np.arctan(n/d)
	elif n>=0 and d<0:
        	alpha=np.pi-np.arctan(n/abs(d))
	elif n<0 and d<0:
        	alpha=np.pi+np.arctan(abs(n)/abs(d))
        	#alpha=-np.pi+np.arctan(abs(n)/abs(d))
        elif n<0 and d>0:
                alpha=2.*np.pi-np.arctan(abs(n)/d)
                #alpha=-np.arctan(abs(n)/d)


	n=np.sin(theta1)*np.sin(phi1-phi2)
	d=np.sin(theta2)*np.cos(theta1) - np.sin(theta1)*np.cos(theta2)*np.cos(phi2-phi1)

        if n>=0 and d>=0 :
		gamma=np.arctan(n/d)
	elif n>=0 and d<0:
        	gamma=np.pi-np.arctan(n/abs(d))
	elif n<0 and d<0:
        	gamma=np.pi+np.arctan(abs(n)/abs(d))
        	#gamma=-np.pi+np.arctan(abs(n)/abs(d))
        elif n<0 and d>0:
                gamma=2.*np.pi-np.arctan(abs(n)/d)	
                #gamma=-np.arctan(abs(n)/d)	

	gamma=2.*np.pi-gamma

	return alpha,beta,gamma	

def return_euler_pipi(theta1,phi1,theta2,phi2):
	
	n = np.sin(theta1)*np.sin(theta2)*np.cos(phi1-phi2) + np.cos(theta1)*np.cos(theta2)
	beta = np.arccos(n)

	n = np.sin(theta2)*np.sin(abs(phi1-phi2))
	d = np.sin(theta2)*np.cos(theta1)*np.cos(phi1-phi2) - np.sin(theta1)*np.cos(theta2)
	r=n/d
	alpha=np.sign(r)*np.arctan(abs(r))

	n=np.sin(theta1)*np.sin(abs(phi1-phi2))
	d=np.sin(theta2)*np.cos(theta1) - np.sin(theta1)*np.cos(theta2)*np.cos(phi2-phi1)
	r=n/d
	gamma=np.sign(r)*np.arctan(abs(r))

	return alpha,beta,gamma


def return_euler_sin(theta1,phi1,theta2,phi2):
	
	n = np.sin(theta1)*np.sin(theta2)*np.cos(phi1-phi2) + np.cos(theta1)*np.cos(theta2)
	beta = np.arccos(n)

	n = np.sin(theta2)*np.sin(abs(phi1-phi2))
	d = np.sin(beta)
	r=n/d
	alpha=np.sign(r)*np.arcsin(abs(r))

	n=np.sin(theta1)*np.sin(abs(phi1-phi2))
	d=np.sin(beta)
	r=n/d
	gamma=-np.sign(r)*np.arcsin(abs(r))

	return alpha,beta,gamma
	

def euler2coord(alpha,beta,gamma,theta1,phi1):

	n=np.cos(theta1)*np.cos(beta) + np.sin(theta1)*np.sin(beta)*np.cos(phi1-alpha)
	theta2=np.arccos(n)

	n=np.sin(phi1-alpha)
	d=np.cos(phi1-alpha)*np.cos(beta) - np.cos(theta1)*np.sin(beta)/np.sin(theta1)

	if n>=0 and d>=0 :
		phi2=np.arctan(n/d)
	elif n>=0 and d<0:
        	phi2=np.pi-np.arctan(n/abs(d))
	elif n<0 and d<0:
        	phi2=np.pi+np.arctan(abs(n)/abs(d))
        elif n<0 and d>0:
                phi2=2.*np.pi-np.arctan(abs(n)/d)	

	phi2=phi2-gamma
	return theta2,phi2
