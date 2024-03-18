import torch
import torch.nn as nn
import numpy as np


class VideoFiringRateEncoder(nn.Module):
    def __init__(
        self,
        core,
        readout,
        *,
        shifter=None,
        modulator=None,
        elu_offset=0.0,
        nonlinearity_type="elu",
        nonlinearity_config=None,
        use_gru=False,
        gru_module=None,
        twoD_core=False,
    ):
        """
        An Encoder that wraps the core, readout and optionally a shifter amd modulator into one model.
        The output is one positive value that can be interpreted as a firing rate, for example for a Poisson distribution.
        Args:
            core (nn.Module): Core model. Refer to neuralpredictors.layers.cores
            readout (nn.ModuleDict): MultiReadout model. Refer to neuralpredictors.layers.readouts
            elu_offset (float): Offset value in the final elu non-linearity. Defaults to 0.
            shifter (optional[nn.ModuleDict]): Shifter network. Refer to neuralpredictors.layers.shifters. Defaults to None.
            modulator (optional[nn.ModuleDict]): Modulator network. Modulator networks are not implemented atm (24/06/2021). Defaults to None.
            nonlinearity (str): Non-linearity type to use. Defaults to 'elu'.
            nonlinearity_config (optional[dict]): Non-linearity configuration. Defaults to None.
            use_gru (boolean) : specifies if there is some module, which should be called between core and readouts
            gru_module (nn.Module) : the module, which should be called between core and readouts
            twoD_core (boolean) : specifies if the core is 2 or 3 dimensinal to change the input respectively
        """
        super().__init__()
        self.core = core
        self.readout = readout
        self.shifter = shifter
        self.modulator = modulator
        self.offset = elu_offset
        self.use_gru = use_gru
        self.gru_module = gru_module

        if nonlinearity_type != "elu" and not np.isclose(elu_offset, 0.0):
            warnings.warn(
                "If `nonlinearity_type` is not 'elu', `elu_offset` will be ignored"
            )
        if nonlinearity_type == "elu":
            self.nonlinearity_fn = nn.ELU()
        elif nonlinearity_type == "identity":
            self.nonlinearity_fn = nn.Identity()
        else:
            self.nonlinearity_fn = activations.__dict__[nonlinearity_type](
                **nonlinearity_config if nonlinearity_config else {}
            )
        self.nonlinearity_type = nonlinearity_type
        self.twoD_core = twoD_core

        self.rate2fluo = Rate2Fluo(n_neurons=3175).to('cuda')
    def forward(
        self,
        inputs,
        *args,
        targets=None,
        data_key=None,
        behavior=None,
        pupil_center=None,
        trial_idx=None,
        shift=None,
        detach_core=False,
        **kwargs,
    ):
        if self.twoD_core:
            batch_size = inputs.shape[0]
            time_points = inputs.shape[1]
            inputs = torch.transpose(inputs, 1, 2)
            inputs = inputs.reshape(((-1,) + inputs.size()[2:]))

        x = self.core(inputs)
        if detach_core:
            x = x.detach()

        if self.use_gru:
            if self.twoD_core:
                x = x.reshape(((batch_size, -1) + x.size()[1:]))
                x = torch.transpose(x, 1, 2)
            x = self.gru_module(x)
            if isinstance(x, list):
                x = x[-1]

        x = torch.transpose(x, 1, 2)
        batch_size = x.shape[0]
        time_points = x.shape[1]

        if self.shifter:
            if pupil_center is None:
                raise ValueError("pupil_center is not given")
            pupil_center = pupil_center[:, :, -time_points:]
            pupil_center = torch.transpose(pupil_center, 1, 2)
            pupil_center = pupil_center.reshape(((-1,) + pupil_center.size()[2:]))
            shift = self.shifter[data_key](pupil_center, trial_idx)

        x = x.reshape(((-1,) + x.size()[2:]))
        x = self.readout(x, data_key=data_key, shift=shift, **kwargs)

        if self.modulator:
            if behavior is None:
                raise ValueError("behavior is not given")
            x = self.modulator[data_key](x, behavior=behavior)

        if self.nonlinearity_type == "elu":
            x = self.nonlinearity_fn(x + self.offset) + 1
        else:
            x = self.nonlinearity_fn(x)

        x = x.reshape(((batch_size, time_points) + x.size()[1:]))
        # return x
        n_neurons = x.shape[2]
        # self.rate2fluo = Rate2Fluo(time_points=time_points, n_neurons=n_neurons, batch_size=batch_size).to('cuda')

        x = self.rate2fluo(lmbda = x.permute(2, 0, 1).contiguous().view(-1, 1, time_points), time_points=time_points, batch_size=batch_size)

        # x = x.view(n_neurons, batch_size, time_points).permute(1, 2, 0)
        return x

    def regularizer(
        self, data_key=None, reduction="sum", average=None, detach_core=False
    ):
        reg = (
            self.core.regularizer().detach() if detach_core else self.core.regularizer()
        )
        reg = reg + self.readout.regularizer(
            data_key=data_key, reduction=reduction, average=average
        )
        if self.shifter:
            reg += self.shifter.regularizer(data_key=data_key)
        if self.modulator:
            reg += self.modulator.regularizer(data_key=data_key)
        return reg

class Rate2Fluo(nn.Module):
    def __init__(self, n_neurons=3175):
        super().__init__()
        # self.time_points = time_points
        self.n_neurons = n_neurons
        # self.batch_size = batch_size
        self.alpha = nn.Parameter(torch.rand(n_neurons))
        self.beta = nn.Parameter(torch.rand(n_neurons))

        nn.init.constant_(self.alpha, 1)
        nn.init.constant_(self.beta, 0)

    def forward(self, lmbda, time_points, batch_size):

        Ct = nn.functional.conv1d(lmbda.float(), gcamp8.view(1, 1, -1).float(), padding = 20)

        Ct = Ct[:, :, :time_points]

        Ct = Ct.view(self.n_neurons, batch_size, time_points).permute(1, 2, 0)

        Ft = torch.einsum('ijk,k->ijk', Ct, self.alpha) + self.beta
        # Ft = self.alpha * Ct + self.beta

        return Ft
    
num_timesteps = 300
time_per_timestep = 33.33
timestep_1 = 20
tau = 149

time_values = np.arange(0, num_timesteps * time_per_timestep, time_per_timestep)

decay_values = np.exp(-(time_values - timestep_1 * time_per_timestep) / tau)
decay_values[timestep_1:] *= 1
decay_values[:timestep_1] *= 0

gcamp8_np_flip = np.flip(decay_values[20:40])
gcamp8 = torch.from_numpy(gcamp8_np_flip.copy()).to('cuda')