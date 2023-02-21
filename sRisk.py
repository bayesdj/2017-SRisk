import numpy as np
import pandas as pd
from numpy imort log, pi, array, zeros, ones, diag, sqrt, exp, random
from numpy.linalg import det, inv, solve
from scipy.optimize import fmin_slsqp
from numba import jit
from datetime import datetime
import os, pickle
#%%
eps = np.finfo(float).eps
small = 1e-5

def getSRisk(d1, days=252,S=1e6):
    x = estPast(d1=d1,days=days)
    s = sim(x,S=S)
    s.cleanup()
    path = '.\\Results\\' + d1.strftime('%Y-%m-%d') + '.obj'
    save(s,path)
    return s.srisk

def save(obj,name):
    with open(name,'wb') as myx:
        return pickle.load(myx)
    
def load(name):
    with open(name,'rb') as myx:
        return pickle.load(myx)
    
def standardize(M):
    P = diag(1/sqrt(diag(M)))
    return P@M@P

@jit
def gjr_garch_likelihood(parameters,data,sigma2,out=None):
    ''' returns negative loglikelihood for GJRGARCH(1,1,1) model '''
    omega,alpha,gamma,beta = parameters
    T = data.shape[0]
    eps = data
    # Data and sigma2 are T by 1 vectors
    for t in range(1,T):
        sigma2[t] = (omega + alpha * eps[t-1]**2 + gamma * eps[t-1]**2 * (eps[t-1]<0)
                    + beta*sigma2[t-1])
        
    logliks = 0.5*(log(2*pi) + log(sigma2) + eps**2/sigma2)
    loglik = logliks.mean()
    
    if out is None:
        return loglik
    else:
        return loglik, logliks, sigma2.copy()
    
def gjr_constraint(parameters, data, sigma2, out=None):
    '''Constraint that alpha+gamma/2+beta<=1'''
    alpha,gamma,beta = parameters[1:]
    return 1 - alpha + gamma*0.5 - beta - small

def getTheta(series):
    x = np.asarray(series.dropna())
    var = x.var()
    v0 = array([var*0.01,0.03,0.09,0.9])
    founds = [(eps, 4*var), (0,1), (0,1), (0,1)]
    sig2 = ones(x.shape[0]) * var
    theta,fx,its,imode,smode = fmin_slsqp(gjr_garch_likelihood, v0, f_ieqcons=gjr_constraint, bounds= bounds,
                                          args = (x,sig2),iprint=0,full_output=True)
    if imode != 0:
        raise(Exception('gjr garch optimization failed ' + smode + ' / Current function value: ' + str(fx)))
    _,_,sig2 = gjr_garch_likelihood(theta,x,sig2,out=1)
    return pd.Series(np.concatenate((theta,sig2)))

class estPast():
    def __init__(self,d1=datetime(2017,5,5),days=2500,days0=255,path=None,use=None):
        self.path = os.getcwd() if path is None else path
        self.h5 = self.path + '\\Data\\Data.h5'
        self.d1 = d1
        self.getPrice(use,d1,days,days0)
        self.garchApply()
        self.theta0 = array([0.06,0.9])
        self.cdcc_opt()
        
    def getPrice(self,use,d1,days,days0):
        with pd.HDFStore(self.h5) as H5:
            Price = H5['p'].loc[:d1]
            pmkt = H5['pmkt'].loc[:d1]
            
        if use is None:
            P = Price.iloc[-days-3:].dropna(axis=1)
            use = P.columns
            
        P = Price[use].iloc[-days-days0:].join(Pmkt)
        r = log(P/P.shift(1))*100
        self.returns = r.iloc[-days:].dropna()
        self.corr0 = r.iloc[:-days].corr().values[-1,:-1]
        self.use = use
        
    def garchApply(self):
        df = self.returns.apply(getTheta)
        Theta = df.iloc[:4]
        Theta.index = ['omega','alpha','gamma','beta']
        var = df.iloc[4:].as_matrix()
        T,k = var.shape; k-=1
        e = self.returns.as_matrix()/sqrt(var)
        Rbar = standardize((e.T@e)/T)
        e1 = e[:,:-1]; e2 = np.tile(e[:,-1],(k,1)).T
        E = array(tuple(zip(e1,e2))).transpose(0,2,1)
        S = E[:,:,:,None]
        self.thetaGarch = Theta
        self.varGarch = var
        self.thi0 = Rbar[-1,:-1]
        self.T = T
        self.k = k
        self.eGarch = e
        self.eye = np.tile(np.eye(2),(k,1,1))
        self.eye1 = 1-self.eye
        self.Q0 = self.eye1*self.corr0[:,None,None] + self.eye
        self.S = S
        self.St = S.transpose(0,1,3,2)
        self.R = np.tile(np.eye(2),(T-1,k,1,1))
        #self.SS = S@self.St
        
    def cdcc_opt(self,theta0=None,thi0=None,iprint=1):
        theta0 = self.theta0 if theta0 is None else theta0
        theta1,Thi0 = self.cdcc_opt2(theta0,iprint=0)
        theta2 = self.cdcc_opt1(theta1,iprint=iprint)
        the1 = theta2[2:]
        theta3,Thi1 = sefl.cdcc_opt2(theta2[:2],thi1,iprint=0)
        RHO,Q = self.cdcc_cle(theta3,Thi1,out=1)
        self.thi1 = thi1
        self.thetaDcc = theta3
        self.Qt = Q
        self.RHO = RHO
        print(self.d1.strftime('%Y-%m-%d') + ' cdcc_opt success')
        
    def cdcc_opt1(self,theta,thi=None,iprint=1):
        thi = self.thi0 if thi is None else thi
        bound = [(0,1),(0,1)] + [(-np.inf,np.inf) for i in range(self.k)]
        theta = np.concatenate((theta,thi))
        est,fx,its,imode,smode = fmin_slsqp(self.cdcc_cld,theta,bound=bound,f_ieqcons=self.constraint_cle,
                                            iprint=iprint,full_output=True)
        if imode !=0:
            raise(Exception('cdcc_opt1 failed: ' + smode + ' / Current function value: ' + str(fx)))
        return est
    
    def cdcc_opt2(self,theta,thi=None,iprint=0):
        thi = self.thi0 if thi is None else thi
        bound = ((0,1),(0,1))
        Thi = self.eye1*thi[:,None,None]+self.eye
        est,fx,its,imode,smode = fmin_slsqp(self.cdcc_cle2,theta,bounds=bound,f_ieqcons=self.constraint_cle,
                                            args=(Thi,),iprint=iprint,full_output=True)
        if imode !=0:
            raise(Exception('cdcc_opt2 failed: ' + smode + ' / Current function value: ' + str(fx)))
        return est,Thi
    
    def constraint_cle(self,theta,Thi=None,out=None):
        alpha,beta = theta[:2]
        return 1-alpha-beta-small
    
    @jit
    def cdcc_cle2(self,theta,Thi,out=None):
        alpoha,beta = theta
        Omega = (1-alpha-beta)*Thi
        P = self.eye
        Q = self.Q0
        R = self.R.copy()
        for t in range(self.T-1):
            Q = Omega + alpha*P@self.S[t]@self.St[t]@P + beta*Q
            P = sqrt(Q*self.eye)
            invP = (1/(P+self.eye1))*self.eye
            R[t] = invP@Q@invP
        if out is None:
            Lik = log(det(R)) + (self.St[1:]@solve(R,self.S[1:]))[:,:,0,0] # St@invR@S
            return Lik.mean()
        else:
            RHO = R[:,:,0,1]
            RHO = np.concatenate((self.corr0[None,:],RHO))
            return RHO,Q
        
    @jit
    def cdcc_cle(self,theta):
        alpha,beta = theta[:2]
        thi = theta[2:]
        Thi = self.eye1*thi[:,None,None] + self.eye
        Omega = (1-alpha-beta)*Thi
        P = self.eye
        Q = self.Q0
        R = self.R.copy()
        for t in range(self.T-1):
            Q = Omega + alpha*P@self.S[t]@self.St[t]@P + beta*Q
            P = sqrt(Q*self.eye)
            invP = (1/(P+self.eye1))*self.eye
            R[t] = invP@Q@invP
        Lik = log(det(R)) + (self.St[1:]@solve(R,self.S[1:]))[:,:,0,0] 
        return Lik.mean()
    
class sim():
    def __init__(self,estimate,h=22,S=1e6,C=0.1,k=0.08,path=None):
        self.x = estimate
        self.h = int(h)
        self.S = int(S)
        self.C = -C
        self.bootstrapSim()
        self.computeSRisk(k)
        
    def cleanup(self):
        del self.x.R
        del self.x.eye
        del self.x.eye1
        del self.x.S
        del self.x.St
        del self.x.RHO
        del self.x.returns
        
    def computeSRisk(self,k):
        use = self.x.use
        d1 = self.x.d1
        with pd.HDFStore(self.x.H5) as H5:
            lvg = H5['lvg'].loc[:d1,use].iloc[-1]
            w = H5['mv'].loc[:d1,use].iloc[-1]
        self.srisk = w*(k*lvg + (1-k)*self.lrmes -1)
        
    def bootstrapSim(self):
        x = self.x
        RHO = x.RHO
        em = x.eGarch[:,[-1]]
        ei = (x.eGarch[:,:-1]-RHO*em)/sqrt(1-RHO*RHO)
        Em = random.choice(em.flatten(),size=(self.h+1,self.S))
        Em[0,:] = em[-1,-1]
        theta = x.thetaGarch['zzz']; theta['vT'] = x.varGarch[-1,-1]
        theta['rT'] = x.returns.iloc[-1,-1]; theta['C'] = self.C
        Rm Vm, em = self.simGJR(Em,theta)
        s = Rm.shape[1]
        k = self.x.k; h = self.h
        Ei = np.apply_along_axis(random.choice,0,ei,size=h*s).reshape((h,s,k)).transpose((0,2,1))
        Em = np.tile(em,(k,1,1)).transpose((1,0,2))
        Theta = x.thetaGarch[x.use].T
        Theta['vT'] = x.varGarch[-1,:-1]
        Theta['rT'] = x.returns[x.use].iloc[-1]
        omega,alpha,gamma,beta,v,r = map(lambda x: Theta[x][:,None],Theta.columns)
        Thi = np.tile(x.thi1,(1,1,1,1)).transpose((3,0,1,2))
        eye = np.tile(np.eye(2),(k,s,1,1))
        eye1 = 1-eye
        a,b = x.thetaDcc
        Omega = (1-a-b)*(eye1*Thi+eye)
        Q = eye@x.QT[:,None,:]
        P = sqrt(Q*eye)
        invP = (1/(P+eye1))*eye
        corr = (invP@Q@invP)[:,:,0,1]
        Corr = zeros((h+1,k,s)); Corr[0] = corr
        Ret = zeros((h,k,s))
        e = x.eGarch[-1,:-1,None]@np.ones((1,s))
        S = array(tuple(zip(Em[0],e))).transpose((0,2,1))[:,:,:,None]
        self.loopSim(h,Q,Omega,a,P,S,b,eye,eye1,Corr,omega,alpha,gamma,r,e,beta,v,Vm,Ei,Ret,Rm,Em)
        
    @jit
    def loopSim(self,h,Q,Omega,a,P,S,b,eye,eye1,Corr,omega,alpha,gamma,r,e,beta,v,Vm,Ei,Ret,Rm,Em):
        use = self.x.use
        for t in range(h):
            Q = Omega + a*P@S@S.transpose(0,1,3,2)@P + b*Q
            P = sqrt(Q*eye)
            invP = (1/(P+eye1))*eye
            corr = (invP@Q@invP)[:,:,0,1]
            Corr[t+1] = corr
            v = omega + (alpha+gamma*(r<0))*e**2 + beta*v
            mu = Rm[t]*corr*sqrt(v/Vm[t])
            sig = sqrt(v*(1-corr**2))
            r = mu + Ei[t]*sig
            Ret[t] = r
            e = r/sqrt(v)
            S = array(tuple(zip(Em[t+1],e))).transpose((0,2,1))[:,:,:,None]
            
        lrmes = (exp(0.01*Ret.sum(axis=0))-1).mean(axis=1)
        self.Corr = pd.DataFrame(Corr.mean(axis=2),columns=use)
        self.lrmes = pd.Series(-lrmes,index=use)
        
    @jit
    def simGJR(self,noise,theta):
        omega,alpha,gamma,beta,v,r,C = theta
        h,s = noise.shape; h -= 1
        R = zeros((h,s))
        V = zeros((h,s))
        for t in range(h):
            v = omega + (alpha+gamma*(r<0))*noise[t]**t + beta*v
            r = noise[t+1]*sqrt(v)
            R[t,:] = r
            V[t,:] = v
        cumR = exp(R.sum(axis=0)*0.01)-1
        idSystemic = (cumR<C)
        return R[:,idSystemic], V[:,idSystemic], noise[:,idSystemic]