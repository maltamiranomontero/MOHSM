from mogptk.dataset import DataSet
from mogptk.model import Model, Exact, logger
from mogptk.gpr import MixtureKernel
from mogptk.plot import plot_spectrum
from kernel import MultiOutputHarmonizableSpectralKernel

class MOHSM(Model):
    def __init__(self, dataset, P=1, Q=1, model=Exact(), mean=None, center = True, name="MOHSM"):
        if not isinstance(dataset, DataSet):
            dataset = DataSet(dataset)
        dataset.rescale_x()

        spectral = MultiOutputHarmonizableSpectralKernel(
            output_dims=dataset.get_output_dims(),
            input_dims=dataset.get_input_dims()[0],
        )
        kernel_p = MixtureKernel(spectral, Q)
        
        kernel = MixtureKernel(kernel_p,P)

        nyquist = dataset.get_nyquist_estimation()

        super(MOHSM, self).__init__(dataset, kernel, model, mean, name)
        self.Q = Q
        self.P = P
 