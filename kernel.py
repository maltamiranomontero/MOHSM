from mogptk.gpr import MultiOutputKernel, Parameter, config

class MultiOutputHarmonizableSpectralKernel(MultiOutputKernel):
    def __init__(self, output_dims, input_dims, active_dims=None, name="MOHSM"):
        super(MultiOutputHarmonizableSpectralKernel, self).__init__(output_dims, input_dims, active_dims, name)

        # TODO: incorporate mixtures?
        # TODO: allow different input_dims per channel
        magnitude = torch.rand(output_dims)
        
        mean = torch.rand(output_dims, input_dims)
        
        variance = torch.rand(output_dims, input_dims)
        
        lengthscale = torch.zeros(output_dims)
        
        delay = torch.zeros(output_dims, input_dims)
        
        phase = torch.zeros(output_dims)
        
        center = torch.zeros(output_dims, input_dims)

        self.input_dims = input_dims
        
        self.magnitude = Parameter(magnitude, lower=config.positive_minimum)
        
        self.mean = Parameter(mean, lower=config.positive_minimum)
        
        self.variance = Parameter(variance, lower=config.positive_minimum)
        
        self.lengthscale = Parameter(lengthscale, lower=config.positive_minimum)
        
        if 1 < output_dims:
            self.delay = Parameter(delay)
            
            self.phase = Parameter(phase)
            
        self.twopi = np.power(2.0*np.pi,float(self.input_dims)/2.0)
        
        self.center = Parameter(center)
        
    def avg(self, X1, X2=None):
        # X1 is NxD, X2 is MxD, then ret is NxMxD
        if X2 is None:
            X2 = X1
        return (X1.unsqueeze(1) + X2)/2
    
    def Ksub(self, i, j, X1, X2=None):
        # X has shape (data_points,input_dims)
        tau = self.distance(X1,X2)  # NxMxD
        avg = self.avg(X1,X2)  # NxMxD
        
        if i == j:
            variance = self.variance()[i]
            lengthscale = self.lengthscale()[i]**2
            
            alpha = self.magnitude()[i]**2 * self.twopi * variance_1.prod().sqrt() * lengthscale.sqrt() # scalar
            
            exp_1 = torch.exp(-0.5* torch.tensordot(tau**2, variance, dims=1))# NxM  
            exp_2 = torch.exp(-0.5* ((avg-(self.center()[i]))**2) * lengthscale**-1) # NxM
            
            cos = torch.cos(2.0*np.pi * torch.tensordot(tau, self.mean()[i], dims=1))# NxM
            
            return alpha * exp_1 * cos * exp_2 
        
        else:
            inv_variances = 1.0/(self.variance()[i] + self.variance()[j])  # D
            inv_lengthscale = 1.0/(self.lengthscale()[i]**2 + self.lengthscale()[j]**2)  # D

            diff_mean = self.mean()[i] - self.mean()[j]  # D
            
            magnitude = self.magnitude_1()[i]*self.magnitude_1()[j]*torch.exp(-np.pi**2 * diff_mean_1.dot(inv_variances*diff_mean))  # scalar

            mean = inv_variances * (self.variance()[i]*self.mean()[j] + self.variance()[j]*self.mean()[i])  # D
            
            variance = 2.0 * self.variance()[i] * inv_variances * self.variance()[j]  # D
            lengthscale = 2.0 * self.lengthscale()[i]**2 * inv_lengthscale * self.lengthscale()[j]**2  # D
            
            delay = self.delay()[i] - self.delay()[j]  # D
            
            phase = self.phase()[i] - self.phase()[j]  # scalar
            
            center = (self.center()[i] + self.center()[j])/2

            alpha = magnitude * self.twopi * variance.prod().sqrt()*lengthscale.sqrt() # scalar
            
            exp_1 = torch.exp(-0.5 * torch.tensordot((tau+delay)**2, variance, dims=1)) # NxM  
            
            exp_2 = torch.exp(-0.5* ((avg-center)**2) * lengthscale**-1) # NxM
            
            cos = torch.cos(2.0*np.pi * torch.tensordot(tau+delay, mean, dims=1)+phase) # NxM
            
            return alpha * exp_1 * cos * exp_2 