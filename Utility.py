import numpy as np

def grad(F,Z,dz):
    '''
    Computes finite difference gradients of formants in latent space
    F = sklearn model (e.g. KNeighbours regressor) trained to predict formant from latent features
    Z = latent features vector
    dz = increment used for finite differencing
    '''
    dim = Z.shape[0]
    #Make list of incremented Zs
    Zps=np.array([Z]*dim)
    Zms=np.array([Z]*dim)
    for i in range(dim):
        Zps[i,i]+=dz
        Zms[i,i]-=dz

    Fps = F.predict(Zps)
    Fms = F.predict(Zms)
    
    return (Fps-Fms)/(2.0*dz)
    
    
def normalize(F_models, Z, F1,F2,alpha=1e-6,kmax=50,tol=25):
    '''
    Function to shift location in latent space to match formants F1 F2 (in Hz)
    F_models = dictionary of sklearn models which predict formants from latent features
                dictionary keys are [1,2] corresponding to F1 and F2
    Z = initial location in latent space
    F1 = target F1 value
    F2 = target F2 value
    alpha = learning rate
    kmax = maximum number of iterations
    tol = tolerance (Hz) which stops search when distnace to targets is less than tol
    '''
    dz=0.05
    dF1=1000.0
    dF2=1000.0
    k=0
    Zn = Z.copy()
    while (np.abs(dF1)>tol or np.abs(dF2)>tol) and k<kmax:
        F1z = F_model[1].predict([Zn])[0]
        F2z = F_model[2].predict([Zn])[0]
        dF1 = F1 - F1z
        dF2 = F2 - F2z
        dZ = alpha*dF1*grad(F_model[1],Zn,dz) 
        dZ += alpha*dF2*grad(F_model[2],Zn,dz)
        Zn += dZ
        k += 1
    return Zn,F1z, F2z