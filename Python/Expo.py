#Métode Numérique TP N°5
#Stage LPC Juillet 2023

#Importation des libs python

#Première partie

import numpy as np
import scipy 
import matplotlib.pyplot as plt
import random as rdm
import time 

#Définition de la plage

xmin,xmax,xbins = 0,50,120

#Définition des paramètres

lam,B = 0.6,1

#Définition des gaussiennes

def expo(x,lam,B):
	return(B*lam*np.exp(-lam*x))

#Génération de données avec numpy

def Numpy_Expo(N,lam,B):
	return(B*np.random.exponential(1/lam,N))

#Génération de la loi gaussienne Monte Carlo
def MC_Expo(N,lam,B):
    ymax = scipy.optimize.fminbound(lambda x: expo(x,lam,B), xmin, xmax)
    print(ymax)
    lx = []
    i = 0
    while i < N:
        x, y = rdm.uniform(xmin, xmax), rdm.uniform(0, ymax)
        if y <= expo(x,lam,B):
            lx.append(x)
            i += 1
    return lx

#Génération de la loi gaussienne avec la fonction de répartition inverse

def Inverse_Expo(N,lam,B):
	lx,u = [],[]
	for i in range(0,N):
		u.append(rdm.uniform(0,1))
	for x in u:
		lx.append(-B*np.log(1-x)/(lam))
	return(lx)

#Calcul des complexités

def Complexite_Expo(N,lam,B):
	l1,l2,l3 = [],[],[]
	for n in range(1,N,100):
		start = time.time()
		Numpy_Expo(n,lam,B)
		stop = time.time()
		l1.append(stop-start)
		print("1",n,stop-start)
	for n in range(0,N,100):
		start = time.time()
		MC_Expo(n,lam,B)
		stop = time.time()
		l2.append(stop-start)
		print("2",n,stop-start)
	for n in range(0,N,100):
		start = time.time()
		Inverse_Expo(n,lam,B)
		stop = time.time()
		l3.append(stop-start)
		print("3",n,stop-start)

	plt.figure(99)
	plt.plot(l1)
	plt.plot(l2)
	plt.plot(l3)
	plt.legend(['Méthode Numpy', 'Méthode Monte Carlo', 'Méthode de la fonction de répartition inverse'])
	plt.show()

#Calcul des valeurs moyennes et des variances

def Calcul_Expo(N,lam,B):
	l1,l2,l3 = Numpy_Expo(N,lam,B),MC_Expo(N,lam,B),Inverse_Expo(N,lam,B)
	m1,m2,m3,v1,v2,v3 = 0,0,0,0,0,0
	for a,b,c in zip(l1,l2,l3):
		m1,m2,m3 = m1+a,m2+b,m3+b
	m1,m2,m3 = m1/N,m2/N,m3/N
	for a,b,c in zip(l1,l2,l3):
		v1,v2,v3 = v1 + (a-m1)**2,v2 + (a-m2)**2,v3 + (a-m3)**2
	v1,v2,v3=v1/N,v2/N,v3/N
	return((m1,v1),(m2,v2),(m3,v3))
	
#Tracer des histogrammes

def Trace_Expo(N,lam,B):
	l1,l2,l3 = Numpy_Expo(N,lam,B),MC_Expo(N,lam,B),Inverse_Expo(N,lam,B)
	plt.figure(31)
	plt.hist(l1,xbins)
	plt.legend(['Méthode Numpy'])
	plt.figure(41)
	plt.hist(l2,xbins)
	plt.legend(['Méthode Monte Carlo'])
	plt.figure(59)
	plt.hist(l3,xbins)
	plt.legend(['Méthode de la fonction de répartition inverse'])
	plt.show()



