import os
from functools import partial

import numpy as np
import torch
import wandb
from neuralpredictors.measures import modules
from neuralpredictors.training import LongCycler, early_stopping
from nnfabrik.utility.nn_helpers import set_random_seed
from tqdm import tqdm

from ..utility import scores
from ..utility.scores import get_correlations, get_poisson_loss


## TODO - add loading from checkpoints in case of train stop
def standard_trainer(
    model,
    dataloaders,
    seed,
    avg_loss=False,
    scale_loss=True,
    loss_function="PoissonLoss",
    stop_function="get_correlations",
    loss_accum_batch_n=None,
    device="cuda",
    verbose=True,
    interval=1,
    patience=5,
    epoch=0,
    lr_init=0.005,
    max_iter=200,
    maximize=True,
    tolerance=1e-6,
    restore_best=True,
    lr_decay_steps=3,
    lr_decay_factor=0.3,
    min_lr=0.0001,
    cb=None,
    detach_core=False,
    core_state_dict=None,
    use_wandb=True,
    wandb_project="factorised_core_parameter_search",
    wandb_entity="movies_parameter_search",
    wandb_name=None,
    wandb_model_config=None,
    wandb_dataset_config=None,
    print_step=1000,
    save_checkpoints=True,
    checkpoint_save_path="local/",
    chpt_save_step=15,
    deeplake_ds=False,
    validation_str = "oracle", # or "validation"
    T_max_epochs = 100,
    optimizer_type = "AdamW",
    **kwargs,
):
    """

    Args:
        model: model to be trained
        dataloaders: dataloaders containing the data to train the model with
        seed: random seed
        avg_loss: whether to average (or sum) the loss over a batch
        scale_loss: whether to scale the loss according to the size of the dataset
        loss_function: loss function to use
        stop_function: the function (metric) that is used to determine the end of the training in early stopping
        loss_accum_batch_n: number of batches to accumulate the loss over
        device: device to run the training on
        verbose: whether to print out a message for each optimizer step
        interval: interval at which objective is evaluated to consider early stopping
        patience: number of times the objective is allowed to not become better before the iterator terminates
        epoch: starting epoch
        lr_init: initial learning rate
        max_iter: maximum number of training iterations
        maximize: whether to maximize or minimize the objective function
        tolerance: tolerance for early stopping
        restore_best: whether to restore the model to the best state after early stopping
        lr_decay_steps: how many times to decay the learning rate after no improvement
        lr_decay_factor: factor to decay the learning rate with
        min_lr: minimum learning rate
        cb: whether to execute callback function
        core_state_dict: if detach_core=True, we would use transfer learning, and the core is borrowed from this state_dict (.pth) file
        **kwargs:

    Returns:

    """

    def full_objective(model, dataloader, data_key, *args, **kwargs):
        loss_scale = (
            np.sqrt(len(dataloader[data_key].dataset) / args[0].shape[0])
            if scale_loss
            else 1.0
        )
        # todo - think how to avoid sum in model.core.regularizer()
        if not isinstance(model.core.regularizer(), tuple):
            regularizers = int(
                not detach_core
            ) * model.core.regularizer() + model.readout.regularizer(data_key)
        else:
            regularizers = int(not detach_core) * sum(
                model.core.regularizer()
            ) + model.readout.regularizer(data_key)
        if deeplake_ds:
            for k in kwargs.keys():
                if k not in ["id", "index"]:
                    kwargs[k] = torch.Tensor(np.asarray(kwargs[k])).to(device)
        model_output = model(args[0].to(device), data_key=data_key, **kwargs)
        time_left = model_output.shape[1]

        original_data = args[1].transpose(2, 1)[:, -time_left:, :].to(device)

        return (
            loss_scale
            * criterion(
                model_output,
                original_data,
            )
            + regularizers
        )

    ##### Model training ####################################################################################################
    model.to(device)
    set_random_seed(seed)
    model.train()

    criterion = getattr(modules, loss_function)(avg=avg_loss)
    stop_closure = partial(
        getattr(scores, stop_function),
        dataloaders=dataloaders[validation_str],
        device=device,
        per_neuron=False,
        avg=True,
        deeplake_ds=deeplake_ds,
    )

    n_iterations = len(LongCycler(dataloaders["train"]))

    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr_init)
    if optimizer_type == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_init)
    elif optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr_init)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max" if maximize else "min",
        factor=lr_decay_factor,
        patience=patience,
        threshold=tolerance,
        min_lr=min_lr,
        verbose=verbose,
        threshold_mode="abs",
    )

    # set the number of iterations over which you would like to accummulate gradients
    optim_step_count = (
        len(dataloaders["train"].keys())
        if loss_accum_batch_n is None
        else loss_accum_batch_n
    )
    print(f"optim_step_count = {optim_step_count}")

    if use_wandb:
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            name=wandb_name,
            # Track hyperparameters and run metadata
            config={
                "learning_rate": lr_init,
                "architecture": wandb_model_config,
                "dataset": wandb_dataset_config,
                "cur_epochs": max_iter,
                "starting epoch": epoch,
                "lr_decay_steps": lr_decay_steps,
                "lr_decay_factor": lr_decay_factor,
                "min_lr": min_lr,
            },
        )

        wandb.define_metric(name="Epoch", hidden=True)
        wandb.define_metric(name="Batch", hidden=True)

    batch_no_tot = 0
    # train over epochs
    for epoch, val_obj in early_stopping(
        model,
        stop_closure,
        interval=interval,
        patience=patience,
        start=epoch,
        max_iter=max_iter,
        maximize=maximize,
        tolerance=tolerance,
        restore_best=restore_best,
        scheduler=scheduler,
        lr_decay_steps=lr_decay_steps,
    ):
        # executes callback function if passed in keyword args
        if cb is not None:
            cb()

        # train over batches
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0
        epoch_val_loss = 0
        for batch_no, (data_key, data) in tqdm(
            enumerate(LongCycler(dataloaders["train"])),
            total=n_iterations,
            desc="Epoch {}".format(epoch),
        ):
            batch_no_tot += 1
            batch_args = list(data)

            if batch_no_tot%n_iterations==0:
                # cycle_start, cycle_number = update_tau_cyclic_increasing(model, epoch, data_key, cycle_start=cycle_start, cycle_number=cycle_number, T_0=20)
                # update_tau_cyclic(model, epoch, data_key, T_cycle=20)
                update_tau_cosine_epoch(model, epoch, data_key, T_max_epochs=T_max_epochs)
            # update_tau_inverse_time(model, batch_no_tot, data_key, interval = n_iterations)
            # print(f"batch_no_tot: {batch_no_tot}, tau: {model.readout[data_key].tau.item()}")

            batch_kwargs = data._asdict() if not isinstance(data, dict) else data

            loss = full_objective(
                model,
                dataloaders["train"],
                data_key,
                *batch_args,
                **batch_kwargs,
                detach_core=detach_core,
            )
            loss = loss/optim_step_count
            loss.backward()

            epoch_loss += loss.detach()
            if (batch_no + 1) % optim_step_count == 0: # TODO: or (batch_no + 1 == len(LongCycler(dataloaders["train"])))
                optimizer.step()

                #                 optimizer.zero_grad(set_to_none=False)
                optimizer.zero_grad(set_to_none=True)
        
        model.eval()
        #yqiu
        #if save_checkpoints:
        #    if epoch % chpt_save_step == 0:
        #        torch.save(
        #            model.state_dict(), f"{checkpoint_save_path}epoch_{epoch}.pth"
        #        )

        ## after - epoch-analysis

        validation_correlation = get_correlations(
            model,
            dataloaders[validation_str],
            device=device,
            as_dict=False,
            per_neuron=False,
            deeplake_ds=deeplake_ds,
        )
        val_loss = full_objective(
            model,
            dataloaders[validation_str],
            data_key,
            *batch_args,
            **batch_kwargs,
            detach_core=detach_core,
        )
        print(
            f"Epoch {epoch}, Batch {batch_no}, Train loss {loss}, Validation loss {val_loss}"
        )
        print(f"EPOCH={epoch}  validation_correlation={validation_correlation}")

        if use_wandb:
            wandb_dict = {
                "Epoch Train loss": epoch_loss,
                #yqiu
                #"Batch": batch_no_tot, 
                #"Epoch": epoch,
                "validation_correlation": validation_correlation,
                "Epoch validation loss": val_loss,
                "Epoch": epoch,
                "tau": model.readout[data_key].tau.item(),
            }
            wandb.log(wandb_dict)
        model.train()

    ##### Model evaluation ####################################################################################################
    model.eval()
    if save_checkpoints:
        #yqiu
        #torch.save(model.state_dict(), f"{checkpoint_save_path}final.pth")
        torch.save(model.state_dict(), f"{checkpoint_save_path}.pth")

    # Compute avg validation and test correlation
    validation_correlation = get_correlations(
        model, dataloaders[validation_str], device=device, as_dict=False, per_neuron=False, deeplake_ds=deeplake_ds,
    )
    print(f"\n\n FINAL validation_correlation {validation_correlation} \n\n")

    output = {}
    output["validation_corr"] = validation_correlation

    score = np.mean(validation_correlation)
    if use_wandb:
        wandb.finish()

    # removing the checkpoints except the last one
    #yqiu, comment these lines
    #to_clean = os.listdir(checkpoint_save_path)
    #for f2c in to_clean:
    #    if "epoch_" in f2c and f2c[-4:] == ".pth":
    #        os.remove(f"{checkpoint_save_path}{f2c}")

    return score, output, model.state_dict()

def update_tau_exponential(model, step, data_key, start_tau=10.0, min_tau=0.5, decay_rate=3e-5, interval=200):
    if step % interval == 0:
        new_tau = max(min_tau, start_tau * torch.exp(torch.tensor(-decay_rate * step)).item())
        model.readout[data_key].tau.copy_(torch.tensor(new_tau, dtype=torch.float32))

def update_tau_inverse_time(model, step, data_key, start_tau=10.0, min_tau=0.5, k=1e-4, interval=200):
    if step % interval == 0:
        new_tau = max(min_tau, start_tau / (1 + k * step))
        model.readout[data_key].tau.copy_(torch.tensor(new_tau, dtype=torch.float32))

def update_tau_cosine_epoch(model, epoch, data_key, T_max_epochs=100, tau_max=10.0, tau_min=0.5):
    t = min(epoch, T_max_epochs)
    cos_input = torch.tensor(torch.pi * t / T_max_epochs)
    new_tau = tau_min + 0.5 * (tau_max - tau_min) * (1 + torch.cos(cos_input))
    model.readout[data_key].tau.copy_(new_tau)

def update_tau_cyclic(model, epoch, data_key, T_cycle=20, tau_max=10.0, tau_min=0.5):
    """
    Cyclic cosine annealing for tau, restarted every T_cycle epochs.
    It is sort of like cosine annealing but every T_cycle epochs
    """
    t_mod = epoch % T_cycle
    cos_input = torch.tensor(torch.pi * t_mod / T_cycle)
    new_tau = tau_min + 0.5 * (tau_max - tau_min) * (1 + torch.cos(cos_input))
    model.readout[data_key].tau.copy_(new_tau)

def update_tau_cyclic_increasing(
    model,
    epoch,
    data_key,
    cycle_start,
    cycle_number,
    T_0=20,
    tau_max=10.0,
    tau_min=0.5,
):
    T_cycle = T_0 * (2 ** cycle_number)
    t = epoch - cycle_start

    if t >= T_cycle:
        # Begin a new cycle
        cycle_number += 1
        cycle_start = epoch
        T_cycle = T_0 * (2 ** cycle_number)
        t = 0

    # Cosine annealing within current cycle
    cos_input = torch.tensor(torch.pi * t / T_cycle)
    new_tau = tau_min + 0.5 * (tau_max - tau_min) * (1 + torch.cos(cos_input))
    model.readout[data_key].tau.copy_(new_tau)

    return cycle_start, cycle_number