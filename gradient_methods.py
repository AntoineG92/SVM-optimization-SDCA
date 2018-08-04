import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import pandas as pd

class SDCA():
    def __init__(self,X,y,gamma,N_epoch):
        self.X=X
        self.y=y
        self.n=len(X)
        self.D=X.shape[1]
        self.gamma=gamma
        self.N_epoch=N_epoch
        self.T=self.N_epoch*self.n
        self.T0=int(self.T/2)
        
    def delta_alpha(self,X,w,y,alpha):
        min_delta = min(1,( 1 - np.dot(X.T,w)*y - self.gamma*alpha)/((np.linalg.norm(X)**2)/(self.lamb*self.n)+self.gamma)+alpha*y)
        return( y*max(0,min_delta) - alpha )
    
    def hinge_loss(self,a,y):
        return( max(0,1-y*a) )

    def hinge_loss_smooth(self,a,y):
        if a>1:
            return 0
        if a<1-self.gamma:
            return(1-a-self.gamma/2)
        return(((1-a)**2)/(2*self.gamma))
    
    def optim_primal(self,X,y,w):
        hinge=0
        for i in range(len(X)):
            if self.gamma==0:
                hinge=hinge+self.hinge_loss(np.dot(w.T,X[i,:]),y[i])
            else:
                hinge=hinge+self.hinge_loss_smooth(np.dot(w.T,X[i,:]),y[i])
        return( hinge/len(X) + 0.5*self.lamb*np.linalg.norm(w)**2)
    
    def optim_dual(self,X,y,alpha):
        temp=0
        sum1=np.dot(alpha.T,y)
        sum2=np.dot(alpha.T,X)
        val2=0.5*(1/(self.lamb*self.n**2))*np.linalg.norm(sum2)**2
        if self.gamma==0:
            val1 = sum1/self.n
        else:
            val1=0
            for i in range(self.n):
                val1=val1+alpha[i]-self.gamma/2*alpha[i]**2
            val1=val1/len(X)
        return( val1 - val2  )

    def SDCA_algo(self,lamb):
        self.lamb=lamb
        X=self.X.copy()
        y=self.y.copy()
        T,T0,D,n=self.T,self.T0,self.D,self.n
        #vectors subject to SDCA minimization
        w=np.zeros(D)
        w_prev=np.zeros(D)
        alpha=np.zeros(n)
        alpha_prev=np.zeros(n)
        w_list=[]
        alpha_list=[]
        e_i=np.identity(n)
        optim_primal_evol=[]
        optim_dual_evol=[]
        dual_gap=1
        epsilon=10**(-3)
        k=0
        #random permutation of x1...xn
        ind_perm=npr.choice(np.arange(0,n),n,replace=False)
        epoch=0
        while ((dual_gap>epsilon) & (epoch<self.N_epoch) ):
            if(k==len(ind_perm)):
                optim_primal_evol.append(self.optim_primal(X,y,w))
                optim_dual_evol.append(self.optim_dual(X,y,alpha))
                dual_gap = self.optim_primal(X,y,w) - self.optim_dual(X,y,alpha)
                ind_perm=npr.choice(np.arange(0,n),n,replace=False)
                epoch+=1
                k=0
            i=ind_perm[k]
            alpha_prev=alpha.copy()
            w_prev=w.copy()
            #compute the gradient
            delta_alpha_t=self.delta_alpha(X[i,:],w_prev,y[i],alpha_prev[i])
            #update alpha
            alpha = alpha_prev + delta_alpha_t*e_i[i,:]
            #update w
            w = w_prev + (1/(self.lamb*n))*delta_alpha_t*X[i,:]
            w_list.append(w)
            alpha_list.append(alpha)
            k=k+1
        #random choice of w between T0 and T
        t_choice = len(w_list)-1
        w_final = w_list[t_choice]
        alpha_final=alpha_list[t_choice]
        return(w_final,alpha_final,optim_primal_evol,optim_dual_evol)

class Pegasos():
    def __init__(self,X,y,T):
        self.T=T
        self.X=X
        self.y=y
        self.n=len(X)
        self.D=X.shape[1]
        #self.lamb=lamb
        
    def hinge_loss(self,a,y):
        return( max(0,1-y*a) )

    
    def optim_primal_i(self,X,y,w,i):
        wx=np.dot(w.T,X[i,:])
        #if self.gamma==0:
        hinge=self.hinge_loss(wx,y[i])
        #else:
            #hinge=self.hinge_loss_smooth(wx,y[i])
        return( hinge + 0.5*self.lamb*np.linalg.norm(w)**2)
        
    def Pegasos_algo(self,lamb):
        self.lamb=lamb
        #loss_pegasos=[]
        prim_optim_pegasos=[]
        y=self.y
        X=self.X
        w=np.zeros(self.D)
        w_prev=np.zeros(self.D)
        w_list=[]
        optim_primal=1
        epsilon = 10**(-3)
        t=1
        while ((optim_primal>epsilon) & (t<self.T)):
        #pick randomly observation i
            i = int(npr.uniform(0,self.n))
            w_prev=w.copy()
            #set the step
            eta = 1/(self.lamb*t)
            if y[i]*np.dot(w,X[i,:]) <1:  #if loss is negative
                w=(1-eta*self.lamb)*w_prev + eta*y[i]*X[i,:]
            else:
                w=(1-eta*self.lamb)*w_prev
                #w=w*( min(1,(1/np.sqrt(self.lamb))/np.linalg.norm(w)) )
            #loss_pegasos.append( self.hinge_loss(np.dot(w.T,X[i,:]),y[i] ) )
            w_list.append(w)
            #w_mean=np.mean(np.array(w_list)
            optim_primal = self.optim_primal_i(X, y, w, i)
            prim_optim_pegasos.append( optim_primal )
            t=t+1
        return(w,w_list,prim_optim_pegasos)
        
    
    