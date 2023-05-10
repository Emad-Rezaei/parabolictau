import pandas as pd
import pandas as pd
import csv
import numpy as np
import math
import matplotlib.pyplot as plt
#parameters
m0=9.109*1e-31
q=1.6*1e-19
K=1.38*1e-23
Ke=K/q
T=300.00
hbar=1.05*1e-34
a=5.658
rho=5.320*1e3
wop=52*1e12 
Nop=1/(math.exp(hbar*wop/(K*T))-1)
print(f'Nop is {Nop}')
b=7.53*1e10 #ioffe
Dae=3.13*q
Doe=8.21*q*1e10
Dah=5.48*q
Doh=10.45*q*1e10
d0=8.85*1e-12
drl=12.9
drh=10.89
de=drl*d0
z=1
ctau=10 #constant relaxation time in BoltzWann
Ev=6.8116
Eg=1.317
Ec=Ev+Eg
Egexp=1.317
shf=Egexp-Eg
mc=0.61*m0
mv=0.89*m0
Ef=0.5*(Ec+Ev)
t=np.zeros(1000)
x=1.0 #energy window inside each band [Ev-x: Ec+x]
m=0
q0=6.8*1e6
dpz=0.16
nimp=1e20 #impurity concentration 1/m^3
#number of row with shape
r=np.loadtxt("GaAs_pac.dat", dtype='str').shape[0]
print(f'number of row is {r}')
#number of columns with shape
c=np.loadtxt("GaAs_pac.dat", dtype='str').shape[1]
print(f'number of column is {c}')
g=np.zeros((r,c))
#chemical potential step equals energy step
deltE=g[1,0]-g[0,0]
dos=np.zeros((r,2))
kvec=np.zeros((r,1))
vel=np.zeros((r,1))
wk=np.zeros((r,1))
#Relaxation times
tau=1e40*np.ones((r,6))
p=np.zeros((r,6)) #scattering rate
print(f' rows {dos.shape[0]}')
print(f' columns {dos.shape[1]}')
dos[0,0]=g[0,0]
shidx=math.floor(hbar/deltE*wop/q)
print(f'shidx is {shidx}')
for i in range(0,dos.shape[0]):
    dos[i,0]=dos[0,0]+i*deltE
    if (dos[i,0] <= Ev):
        dos[i,1]=mv*math.sqrt(2*mv*q*(Ev-dos[i,0]))/(math.pi**2*hbar**3)
    elif (dos[i,0]> Ev) and (dos[i,0] <=Ec+shf):
        dos[i,1]=0.00
    else:
        dos[i,1]=mc*math.sqrt(2*mc*q*(-Ec-shf+dos[i,0]))/(math.pi**2*hbar**3)
np.savetxt('anldos.txt',dos, delimiter=' ')
for i in range(0,dos.shape[0]):
    tau[i,0]=dos[i,0]
    if (dos[i,0] <= Ev):
        kvec[i,0]=math.sqrt(2*mv*q*(Ev-dos[i,0]))/hbar
        vel[i,0]=math.sqrt(2*q*(Ev-dos[i,0])/mv)
        tau[i,1]=hbar*b/(Dah**2*K*T*math.pi*dos[i,1])
        if i > shidx:
            tau[i,2]=2*rho*wop/(Doh**2*math.pi)/(dos[i+shidx,1]*Nop+dos[i-shidx,1]*(1+Nop))
        tau[i,4]=(de/q/dpz)**2*2*hbar*b*kvec[i,0]**2/(math.pi*K*T*dos[i,1])/(math.log((4*kvec[i,0]**2+q0**2)/q0**2)-4*kvec[i,0]**2/(4*kvec[i,0]**2+q0**2))
        tau[i,5]=4*hbar*kvec[i,0]**4*de**2/(math.pi*abs(nimp)*z**2*q**4*dos[i,1])/(math.log((4*kvec[i,0]**2+q0**2)/q0**2)-4*kvec[i,0]**2/(4*kvec[i,0]**2+q0**2))
        if kvec[i,0]*vel[i,0]!=0:
            wk[i,0]=2*wop/(kvec[i,0]*vel[i,0])
            if wk[i,0] > 0 and wk[i,0]<1:
                tau[i,3]=4*math.pi*hbar**2*d0*kvec[i,0]/(mc*q**2*wop)/(drl/drh-1)/(Nop*math.log((1+math.sqrt(1+wk[i,0]))/(-1+math.sqrt(1+wk[i,0])))+(Nop+1)*math.log((1+math.sqrt(1-wk[i,0]))/(1-math.sqrt(1-wk[i,0]))))
    elif (dos[i,0]> Ev) and (dos[i,0] <=Ec+shf):
        tau[i,1]=1e40
        tau[i,2]=1e40
        vel[i,0]=0.00
        tau[i,3]=1e40
        tau[i,4]=1e40
        tau[i,5]=1e40
    else:
        kvec[i,0]=math.sqrt(2*mc*q*(dos[i,0]-Ec))/hbar
        vel[i,0]=math.sqrt(2*q*(dos[i,0]-Ec)/mc)
        tau[i,1]=hbar*b/(Dae**2*K*T*math.pi*dos[i,1])
        if i+shidx < dos.shape[0]:
            tau[i,2]=2*rho*wop/(Doe**2*math.pi*(dos[i+shidx,1]*Nop+dos[i-shidx,1]*(Nop+1)))
        tau[i,4]=(de/q/dpz)**2*2*hbar*b*kvec[i,0]**2/(math.pi*K*T*dos[i,1])/(math.log((4*kvec[i,0]**2+q0**2)/q0**2)-4*kvec[i,0]**2/(4*kvec[i,0]**2+q0**2))
        tau[i,5]=4*hbar*kvec[i,0]**4*de**2/(math.pi*abs(nimp)*z**2*q**4*dos[i,1])/(math.log((4*kvec[i,0]**2+q0**2)/q0**2)-4*kvec[i,0]**2/(4*kvec[i,0]**2+q0**2))
        if kvec[i,0]*vel[i,0]!=0:
            wk[i,0]=2*wop/(kvec[i,0]*vel[i,0])
            if wk[i,0] > 0 and wk[i,0]<1:
                tau[i,3]=4*math.pi*hbar**2*d0*kvec[i,0]/(mc*q**2*wop)/(drl/drh-1)/(Nop*math.log((1+math.sqrt(1+wk[i,0]))/(-1+math.sqrt(1+wk[i,0])))+(Nop+1)*math.log((1+math.sqrt(1-wk[i,0]))/(1-math.sqrt(1-wk[i,0]))))

##DOS plot
plt.xlabel('$\mu$-$E_v$ (eV)',fontname="serif", fontsize=14,color='black',weight='normal')
plt.ylabel('DOS ($1/eV$)', fontname="serif", fontsize=14,color='black',weight='bold')
plt.plot(dos[:,0]-Ev,dos[:,1])
plt.xlim([-2.0, 2.0])
plt.ylim([0.0, 1.0])
np.savetxt('anltau.txt',tau, delimiter=' ')


## tau plot
plt.yscale("log")
plt.xlabel('$\mu$-$E_v$ (eV)',fontname="serif", fontsize=14,color='black',weight='normal')
plt.ylabel('P (1/s)', fontname="serif", fontsize=14,color='black',weight='bold')
plt.xlim([-0.50, 0.20])
plt.ylim([1e7,2e15])
plt.plot(tau[:,0]-Ev,1/tau[:,1],color='red',label='ac-a')
plt.plot(tau[:,0]-Ev-0.0,1/tau[:,2],color='green',label='op-a')
plt.plot(tau[:,0]-Ev,1/tau[:,3],color='cyan',label='po-a')
plt.plot(tau[:,0]-Ev,100/tau[:,4],color='dodgerblue',label='pz-a')
plt.plot(tau[:,0]-Ev,10/tau[:,5],color='silver',label='ii-a')
plt.legend()
plt.show()
