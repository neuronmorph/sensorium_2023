from neuralpredictors.layers.readouts import (FullFactorized2d, 
                                              FullGaussian2d,
                                              MultiReadoutSharedParametersBase,
                                              FullGaussian2d_Gumbel_softmax_scheduled_tau,
                                              FullGaussian2d_REINFORCE,
                                              FullGaussian_3d_sample_grid,
                                              FullGaussian2d_learnable_z_tanh,
                                              FullGaussian2d_learnable_z,
                                              FullGaussian2d_Gumbel_softmax,
                                              FullGaussian2d_Gumbel_softmax_learnable_tau,
                                              FullGaussian2d_adaptive_reg)


class MultipleFullGaussian2d(MultiReadoutSharedParametersBase):
    _readout_registry = {
        "unconstrained": FullGaussian2d,
        "gumbel_softmax_scheduled_tau": FullGaussian2d_Gumbel_softmax_scheduled_tau,
        "reinforce": FullGaussian2d_REINFORCE,
        "3d_sample_grid": FullGaussian_3d_sample_grid,
        "learnable_z_tanh": FullGaussian2d_learnable_z_tanh,
        "learnable_z": FullGaussian2d_learnable_z,
        "gumbel_softmax": FullGaussian2d_Gumbel_softmax,
        "gumbel_softmax_learnable_tau": FullGaussian2d_Gumbel_softmax_learnable_tau,
        "adaptive_reg": FullGaussian2d_adaptive_reg,
    }

    def __init__(self, sparse_readout_type="unconstrained", *args, **kwargs):
        if sparse_readout_type not in self._readout_registry:
            raise ValueError(
                f"Unknown readout_type '{sparse_readout_type}'. "
                f"Available: {list(self._readout_registry.keys())}"
            )
        self._base_readout = self._readout_registry[sparse_readout_type]
        
        super().__init__(*args, **kwargs)


class MultipleFullFactorized2d(MultiReadoutSharedParametersBase):
    _base_readout = FullFactorized2d
