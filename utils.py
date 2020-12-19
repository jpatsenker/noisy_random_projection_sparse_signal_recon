import numpy as np
from sklearn.utils import extmath


def sample_sparse_from_unit_ball(n,m,norm):
    """
    Sample an m-sparse vector randomly from the unit ball
    n: int dimension of vector
    m: sparsity of vector
    norm: function, f:R^n -> R, norm associated with desired ball type
    """
    a = np.random.normal(0,1,m)
    x = a/norm(a) * np.random.random()**(1./m)
    z = np.zeros(n)
    pos = np.random.choice(n,m,replace=False)
    z[pos] = x
    return z


def iden(x):
    """
    idk why this is necessary, but you could put a fourier transform instead or something
    """
    return x

def iden_adj(x):
    """
    ok this is one seriously unnecessary...
    """
    return x


def risk(f,f_s,noise_var):
    """
    Statistical risk
    f: candidate vector in R^n
    f_s: true vector in R^n
    noise_var: variance of noise, in R
    """
    n = f.shape[0]
    assert f_s.shape[0] == n
    
    return np.linalg.norm(f-f_s)/float(n) + noise_var

def emp_risk(f, y, Phi):
    """
    Emperical risk
    f: candidate vector in R^n
    y: noisy projection in R^k
    Phi: measurement matrix R^(nxk)
    """
    n = Phi.shape[0]
    k = Phi.shape[1]
    assert f.shape[0] == n
    assert y.shape[0] == k
    return (np.linalg.norm(y - Phi.T.dot(f))**2)/float(k)


def haupt_alg(Phi,y,T,T_adj,init,iters,eps,practical_adj=1.):
    """
    Optimization algorithm detailed in Haupt et. Al. 2006, Signal Reconstruction From Noisy Random Projections
    Phi - measurement matrix
    y - noisy measurement (Phi f + w)
    T - transform for a compressible f
    T_adj - adjoint of transform T
    init - initalizer in R^n
    iters - number of iterations to run optimization scheme
    eps - epsilon from paper
    practical_adj - percent of the theoretical threshold to use (Haupt et Al. claim theoretical bound too conservative in practice)
    """
    
    #Create Phi and get eigenvalue
    P = Phi.T
    n=Phi.shape[0]
    lam = extmath.randomized_svd(P,1)[1][0]**2

    #Theoretical threshold adjusted for practical use
    cons=practical_adj*np.sqrt(2*np.log(2)*np.log(n)/(lam*eps))

    #initialize return lists
    phis = []
    thetas = []
    fs = []
    
    thetas.append(init)
    
    for i in range(iters):
        #Step 1 - gradient descent
        theta_t = thetas[-1]
        phi_t = theta_t + 1./lam*T_adj(P.T.dot(y- P.dot(T(theta_t))))
        
        #Step 2 - mask w/ threshold for l0 projection
        theta_tp1 = phi_t*(np.abs(phi_t)>=cons)
        
        #append to returns
        phis.append(phi_t)
        thetas.append(theta_tp1)
        fs.append(T(theta_tp1))
        
    return phis,thetas,fs

def basis_pursuit(Phi,y,T,T_adj,init,iters,eta,gamma):
    """
    Optimization algorithm designed to solve basis pursuit
    Phi - measurement matrix
    y - noisy measurement (Phi f + w)
    T - transform for a compressible f
    T_adj - adjoint of transform T
    init - initalizer in R^n
    iters - number of iterations to run optimization scheme
    eta - learning rate
    gamma - regularizer ratio
    """
    P = Phi.T
    thetas = []
    fs = []
    
    thetas.append(init)
    
    for i in range(iters):
        theta_t = thetas[-1]
        theta_tp1 = theta_t + eta*(T_adj(P.T.dot(y - P.dot(T(theta_t)))) - gamma*T(np.sign(theta_t)))
        thetas.append(theta_tp1)
        fs.append(T(theta_tp1))
        
    return thetas,fs



def omp(A, y, k, eps=None):
    """
    Orthogonal Matching Pursuit algorithm
    A - measurement matrix
    y - noisy measurement
    k - desired sparsity
    eps - no desired sparsity, run intil low l2 of residual
    """
    residual = y
    idx = []
    if eps == None:
        stopping_condition = lambda: len(idx) == k
    else:
        stopping_condition = lambda: np.inner(residual, residual) <= eps
    while not stopping_condition():
        lam = np.abs(np.dot(residual, A)).argmax()
        idx.append(lam)
        gamma, _, _, _ = np.linalg.lstsq(A[:, idx], y, rcond=None)
        residual = y - np.dot(A[:, idx], gamma)
    
    x=np.zeros(A.shape[1])
    x[idx] = gamma
    return x