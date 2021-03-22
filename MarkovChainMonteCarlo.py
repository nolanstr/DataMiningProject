import numpy as np
import scipy.stats
from numba import njit

class MCMC_MH:
    
    def __init__(self, data, init_mu=20.5e6, init_sigma=1e2, 
                                    iterations=50000, transition_model=None):
            
        self.iterations = iterations
        self.init_mu = init_mu
        self.init_sigma = init_sigma
        self.data = data

        if transition_model is None:
            std_dev_MCMC = 1e2
            self.transition_model = lambda x: [np.random.normal(x[0], 
                std_dev_MCMC*1e1, 1)[0], np.random.normal(x[1], std_dev_MCMC, 1)[0]]
        self.accepted = []
        self.accepted_iterations = []
        self.rejected = []
        self.rejected_iterations = []
        
    
    def __call__(self):
        
        p0 = [self.init_mu, self.init_sigma]

        for i in range(self.iterations):
            print(i) 
            self.p0_like = _compute_log_likelihood(p0[0], p0[1], self.data)
            
            p1 = self.transition_model(p0)
            
            self.p1_like = _compute_log_likelihood(p1[0], p1[1], self.data)
                
            self._logic()
            
            if p1[1] < 0:
                self.accept = False

            if self.accept == True:
                
                self.accepted.append(p1)
                self.accepted_iterations.append(i)
                
                p0 = p1
            
            else:
                
                self.rejected.append(p1)
                self.rejected_iterations.append(i)
                

    def _logic(self):
        
        acceptance_val = np.random.uniform(0,1,1)[0]
        
        if np.exp(self.p1_like - self.p0_like) > acceptance_val:
            
            self.accept = True
        else:
            self.accept = False
#@njit
def _compute_log_likelihood(var, mu, data):
   
    output = mu * data[0]
    ssqe = np.sum((output - data[1]) ** 2)
    t1 = -np.log(2 * np.pi * var) * output.shape[0] / 2.
    t2 = (-1/2.) * (ssqe / var)
    log_like = t1 + t2

    return log_like

