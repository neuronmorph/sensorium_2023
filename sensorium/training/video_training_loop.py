import os
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from neuralpredictors.measures import modules
from neuralpredictors.training import LongCycler, early_stopping
from nnfabrik.utility.nn_helpers import set_random_seed
from tqdm import tqdm

from ..utility import scores
from ..utility.scores import get_correlations, get_poisson_loss


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


def standard_trainer_gs(
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
    wandb_project="sparse readout",
    wandb_entity="standard trainer sparse readout",
    wandb_name=None,
    wandb_model_config=None,
    wandb_dataset_config=None,
    print_step=1000,
    save_checkpoints=True,
    checkpoint_save_path="local/",
    chpt_save_step=15,
    deeplake_ds=False,
    validation_str = "oracle", # or "validation"
    T_max_epochs = 75,
    optimizer_type = "Adam",
    **kwargs,
):
    """
    Based on the sensorium standard trainer. Has the tau scheduler for Sparse Readout Gumbel Softmax. 

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
                update_tau_cosine_epoch(model, epoch, data_key, T_max_epochs=T_max_epochs)

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
                "validation_correlation": validation_correlation,
                "Epoch validation loss": val_loss,
                "Epoch": epoch,
                # "tau": model.readout[data_key].tau.item(),
            }
            wandb.log(wandb_dict)
        model.train()

    ##### Model evaluation ####################################################################################################
    model.eval()
    if save_checkpoints:
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

    return score, output, model.state_dict()


##### Schedulers for temperatture in the sparse readout

def update_tau_exponential(model, step, data_key, start_tau=10.0, min_tau=0.5, decay_rate=3e-5, interval=200):
    if step % interval == 0:
        new_tau = max(min_tau, start_tau * torch.exp(torch.tensor(-decay_rate * step)).item())
        model.readout[data_key].tau.copy_(torch.tensor(new_tau, dtype=torch.float32))

def update_tau_inverse_time(model, step, data_key, start_tau=10.0, min_tau=0.5, k=1e-4, interval=200):
    '''
    Example: update_tau_inverse_time(model, batch_no_tot, data_key, interval = n_iterations)
    '''
    if step % interval == 0:
        new_tau = max(min_tau, start_tau / (1 + k * step))
        model.readout[data_key].tau.copy_(torch.tensor(new_tau, dtype=torch.float32))

def update_tau_cosine_epoch(model, epoch, data_key, T_max_epochs=100, tau_max=10.0, tau_min=0.5):
    """
    Cosine scheduler for temperature parameter in the Sparse Readout.
    """
    t = min(epoch, T_max_epochs)
    cos_input = torch.tensor(torch.pi * t / T_max_epochs)
    new_tau = tau_min + 0.5 * (tau_max - tau_min) * (1 + torch.cos(cos_input))
    model.readout[data_key].tau.copy_(new_tau)

def update_tau_cyclic(model, epoch, data_key, T_cycle=20, tau_max=10.0, tau_min=0.5):
    """
    Cyclic cosine annealing for tau, restarted every T_cycle epochs.
    Cosine annealing but every T_cycle epochs
    Exmaple: update_tau_cyclic(model, epoch, data_key, T_cycle=20)
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
    '''
    Example: cycle_start, cycle_number = update_tau_cyclic_increasing(model, epoch, data_key, cycle_start=cycle_start, cycle_number=cycle_number, T_0=20)
    '''
    T_cycle = T_0 * (2 ** cycle_number)
    t = epoch - cycle_start

    if t >= T_cycle:
        cycle_number += 1
        cycle_start = epoch
        T_cycle = T_0 * (2 ** cycle_number)
        t = 0

    # Cosine annealing within current cycle
    cos_input = torch.tensor(torch.pi * t / T_cycle)
    new_tau = tau_min + 0.5 * (tau_max - tau_min) * (1 + torch.cos(cos_input))
    model.readout[data_key].tau.copy_(new_tau)

    return cycle_start, cycle_number


##### REINFORCE trainer
def standard_trainer_reinforce(
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
    optimizer_type = "Adam",
    baseline_momentum=0.5, # REINFORCE specific param
    lr_init_reinforce=0.08, # REINFORCE specific param
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

    def full_objective_reinforce(
        model,
        dataloader,
        data_key,
        *args,
        **kwargs,
        ):
        """
        Computes the standard loss (Poisson + regularizers) and REINFORCE loss if available.
        baseline_dict: dict storing moving-average baselines per data_key
        """

        model_output = model(args[0].to(device), data_key=data_key, **kwargs)

        time_left = model_output.shape[1]
        original_data = args[1].transpose(2, 1)[:, -time_left:, :].to(device)
        

        loss_scale = (
            np.sqrt(len(dataloader[data_key].dataset) / args[0].shape[0])
            if scale_loss else 1.0
        )

        if not isinstance(model.core.regularizer(), tuple):
            regularizers = int(not detach_core) * model.core.regularizer() + model.readout.regularizer(data_key)
        else:
            regularizers = int(not detach_core) * sum(model.core.regularizer()) + model.readout.regularizer(data_key)

        loss_per_neuron = criterion(model_output, original_data)
        standard_loss = loss_per_neuron.sum()
        

        reinforce_loss = 0.0

        if hasattr(model.readout[data_key], "last_selected_log_probs"):
            with torch.no_grad():
                baseline_dict[data_key] = (
                    baseline_momentum * baseline_dict[data_key]
                    + (1 - baseline_momentum) * loss_per_neuron
                )

            advantage = loss_per_neuron - baseline_dict[data_key] 
            reinforce_loss = (advantage.detach() * model.readout[data_key].last_selected_log_probs).sum() 
            
            beta_entropy = 1  # strength

            if hasattr(model.readout[data_key], "z_logits"):
                z_probs_for_entropy = F.softmax(model.readout[data_key].z_logits, dim=1)  
                entropy_per_neuron = -(z_probs_for_entropy * z_probs_for_entropy.log()).sum(dim=1)  
                factor = entropy_factor(epoch)
                entropy_loss = beta_entropy * factor * entropy_per_neuron.sum()  # minimize entropy if its plus
            else:
                entropy_loss = 0.0
            
            total_loss = loss_scale * (standard_loss + reinforce_loss) + regularizers + entropy_loss

        return total_loss, standard_loss, reinforce_loss

    ##### Model training ####################################################################################################
    model.to(device)
    set_random_seed(seed)
    model.train()

    criterion = getattr(modules, loss_function)(avg=avg_loss, per_neuron=True)
    stop_closure = partial(
        getattr(scores, stop_function),
        dataloaders=dataloaders[validation_str],
        device=device,
        per_neuron=False,
        avg=True,
        deeplake_ds=deeplake_ds,
    )

    n_iterations = len(LongCycler(dataloaders["train"]))

    if optimizer_type == "Adam":
        optimizer = torch.optim.Adam(
            [p for n, p in model.named_parameters() if "z_logits" not in n], lr=lr_init
        )
        optimizer_z = torch.optim.Adam(
            [p for n, p in model.named_parameters() if "z_logits" in n], lr=lr_init_reinforce
        )
    elif optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr_init)

    # --- Baselines per readout ---
    baseline_dict = {dk: torch.zeros(model.readout[dk].outdims, device=device)
                 for dk in dataloaders["train"].keys()}
    baseline_momentum = baseline_momentum

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
    # Example: 
    #   epoch < 20   : LR * 0.1   (small start)
    #   20–50        : LR * 1.0   (increase)
    #   50–80        : LR * 0.2   (decrease)
    #   >=80         : LR * 0.5   (increase again)
    # def z_schedule(epoch: int):
    #     if epoch < 5:
    #         return 0.01
    #     elif epoch < 20:
    #         return 0.5
    #     elif epoch < 25:
    #         return 0.1
    #     elif epoch < 30:
    #         return 0.5
    #     else:
    #         return 0.5
    # def z_schedule(epoch: int):
    #     if epoch < 5:
    #         return 0.01
    #     elif epoch < 10:
    #         return 0.1
    #     elif epoch<15:
    #         return 0.2
    #     elif epoch<20:
    #         return 0.3
    #     elif epoch < 25:
    #         return 0.4
    #     elif epoch < 30:
    #         return 0.5
    #     elif epoch < 35:
    #         return 0.6
    #     elif epoch < 40:
    #         return 0.7
    #     elif epoch < 45:
    #         return 0.8
    #     else:
    #         return 1.0

    # scheduler_z = torch.optim.lr_scheduler.LambdaLR(optimizer_z, lr_lambda=z_schedule)
    
    scheduler_z = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_z, T_max=60, eta_min=0.05)

    def entropy_factor(epoch: int):
        if epoch < 30:
            return 0.0   # no effect
        elif epoch < 45:
            return -10.0  # max entropy
        elif epoch < 50:
            return 0.0   # back to neutral
        else:
            return 1.0  # min entropy

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
        optimizer_z.zero_grad(set_to_none=True)
        epoch_loss = 0
        epoch_standard_loss = 0
        epoch_reinforce_loss = 0
        # epoch_reinforce_loss_all = 0
        epoch_val_loss = 0
        for batch_no, (data_key, data) in tqdm(
            enumerate(LongCycler(dataloaders["train"])),
            total=n_iterations,
            desc="Epoch {}".format(epoch),
        ):
            batch_no_tot += 1
            batch_args = list(data)

            batch_kwargs = data._asdict() if not isinstance(data, dict) else data


            total_loss, standard_loss, reinforce_loss = full_objective_reinforce(
                model,
                dataloaders['train'],
                data_key,
                *batch_args,
                **batch_kwargs
            )

            loss = total_loss / optim_step_count
            loss.backward()
            # print(model.readout[data_key].z_logits.requires_grad)
            # print(model.readout[data_key].z_logits.grad)
            # if (epoch == 1 or epoch%5==0) and batch_no==1:
            #     print(f"EPOCH {epoch} is HEREEEEe, batch_no {batch_no}")
            #     np.save(f"/mnt/vast-nhr/home/nibsupco/superior_colliculus/loss_vs_probs_REINFORCE/try2/epoch_{epoch}_batch_{batch_no}_selected_z.npy", 
            #             model.readout['dynamic29163-4-4-Fluorescence-7b721b-v4a'].z.detach().cpu().numpy())
            #     np.save(f"/mnt/vast-nhr/home/nibsupco/superior_colliculus/loss_vs_probs_REINFORCE/try2/epoch_{epoch}_batch_{batch_no}_reinforce_loss_all.npy", epoch_reinforce_loss_all.detach().cpu().numpy())
            #     np.save(f"/mnt/vast-nhr/home/nibsupco/superior_colliculus/loss_vs_probs_REINFORCE/try2/epoch_{epoch}_batch_{batch_no}_z_logits.npy", 
            #             model.readout['dynamic29163-4-4-Fluorescence-7b721b-v4a'].z_logits.detach().cpu().numpy())
            #     print("requires_grad for z_logits:", model.readout[data_key].z_logits.requires_grad)
            #     print("requires_grad for last_selected_log_probs:", model.readout[data_key].last_selected_log_probs.requires_grad)

            # grad_norms = {}
            # print(lambda_reinforce)
            # param = model.readout[data_key].z_logits
            # for name, loss_term in zip(
            #     ["total_loss", "standard_loss", "reinforce_loss"],
            #     (total_loss, standard_loss, reinforce_loss)
            # ):
            #     # Zero grads
            #     model.zero_grad(set_to_none=True)

            #     # Backward through only this loss term
            #     loss_term.backward(retain_graph=True)

            #     # Measure grad norm for z_logits
            #     if param.grad is None:
            #         grad_norms[name] = 0.0
            #     else:
            #         grad_norms[name] = param.grad.detach().norm().item()
            # print(grad_norms)
            epoch_loss += loss.detach()
            epoch_standard_loss += standard_loss.detach()
            epoch_reinforce_loss += reinforce_loss.detach()
            # epoch_reinforce_loss_all += reinforce_loss_all.detach()
            if (batch_no + 1) % optim_step_count == 0: # TODO: or (batch_no + 1 == len(LongCycler(dataloaders["train"])))
                # torch.nn.utils.clip_grad_norm_([model.readout[data_key].z_logits], max_norm=10.0)
                optimizer.step()
                optimizer_z.step()
                # if (epoch == 1 or epoch%5==0) and batch_no==1:
                #     print(f"EPOCH {epoch} is HEREEEEe, batch_no {batch_no}")
                #     np.save(f"/mnt/vast-nhr/home/nibsupco/superior_colliculus/loss_vs_probs_REINFORCE/try2/epoch_{epoch}_batch_{batch_no}_selected_z.npy", 
                #             model.readout['dynamic29163-4-4-Fluorescence-7b721b-v4a'].z.detach().cpu().numpy())
                #     np.save(f"/mnt/vast-nhr/home/nibsupco/superior_colliculus/loss_vs_probs_REINFORCE/try2/epoch_{epoch}_batch_{batch_no}_reinforce_loss_all.npy", epoch_reinforce_loss_all.detach().cpu().numpy())
                #     np.save(f"/mnt/vast-nhr/home/nibsupco/superior_colliculus/loss_vs_probs_REINFORCE/try2/epoch_{epoch}_batch_{batch_no}_z_logits.npy", 
                #             model.readout['dynamic29163-4-4-Fluorescence-7b721b-v4a'].z_logits.detach().cpu().numpy())
                #     print(model.readout[data_key].z_logits.grad[0,0])
                    # print("requires_grad for z_logits:", model.readout[data_key].z_logits.requires_grad)
                    # print("requires_grad for last_selected_log_probs:", model.readout[data_key].last_selected_log_probs.requires_grad)
                optimizer.zero_grad(set_to_none=True)
                optimizer_z.zero_grad(set_to_none=True)
        
        # if epoch == 1 or epoch%5 == 0:
        #     print(f"EPOCH {epoch} is HEREEEEe")
        #     np.save(f"/mnt/vast-nhr/home/nibsupco/superior_colliculus/loss_vs_probs_REINFORCE/try2/epoch_{epoch}_reinforce_loss_all.npy", epoch_reinforce_loss_all.detach().cpu().numpy())
        #     np.save(f"/mnt/vast-nhr/home/nibsupco/superior_colliculus/loss_vs_probs_REINFORCE/try2/epoch_{epoch}_z_logits_all.npy", 
        #             model.readout['dynamic29163-4-4-Fluorescence-7b721b-v4a'].z_logits.detach().cpu().numpy())
            # print("requires_grad for z_logits:", model.readout[data_key].z_logits.requires_grad)
            # print("requires_grad for last_selected_log_probs:", model.readout[data_key].last_selected_log_probs.requires_grad)
        
        scheduler_z.step()

        model.eval()

        validation_correlation = get_correlations(
            model,
            dataloaders[validation_str],
            device=device,
            as_dict=False,
            per_neuron=False,
            deeplake_ds=deeplake_ds,
        )
        val_loss, val_standard_loss, val_reinforce_loss = full_objective_reinforce(
                model,
                dataloaders[validation_str],
                data_key,
                *batch_args,
                **batch_kwargs,
                detach_core=detach_core,
            )

        print(
            f"Epoch {epoch}, Batch {batch_no}, Train total loss {loss}, Validation loss {val_loss}"
        )
        print(f"EPOCH={epoch}  validation_correlation={validation_correlation}")
        print(
            f"Epoch standard train loss {epoch_standard_loss}, Epoch reinforce train loss {epoch_reinforce_loss}"
        )
        # print(model.readout[data_key].z_logits[:10, :10])
        # print(f"Epoch {epoch} | "
        #   f"LR rest={optimizer.param_groups[0]['lr']:.6f}, "
        #   f"LR z_logits={optimizer_z.param_groups[0]['lr']:.6f}")

        # REINFORCE-specific analysis
        for data_key in dataloaders["train"].keys():
            if hasattr(model.readout[data_key], "z_logits"):
                with torch.no_grad():
                    channel_probs = F.softmax(model.readout[data_key].z_logits, dim=1)
                    entropy = -(channel_probs * torch.log(channel_probs + 1e-8)).sum(dim=1).mean()
                    print(f"  [{data_key}] Channel Selection Entropy: {entropy.item():.4f}")
                    most_popular = channel_probs.argmax(dim=1).bincount(minlength=model.readout[data_key].in_shape[0])
                    top_channels = most_popular.topk(min(10, model.readout[data_key].in_shape[0])).indices.tolist()
                    print(f"  [{data_key}] Most used channels: {top_channels}")



        if use_wandb:
            wandb_dict = {
                "Epoch Train loss": epoch_loss,
                "validation_correlation": validation_correlation,
                "Epoch validation loss": val_loss,
                "Epoch": epoch,
                "entropy": entropy,
                # "tau": model.readout[data_key].tau.item(),
                "Epoch train_poisson_loss": epoch_standard_loss,
                "Epoch train_reinforce_loss":epoch_reinforce_loss,
                "val_poisson_loss": val_standard_loss,
                "val_reinforce_loss":val_reinforce_loss,
                "Learning rate rest":optimizer.param_groups[0]['lr'],
                "Learning rate z_logits":optimizer_z.param_groups[0]['lr']
            }
            wandb.log(wandb_dict)
        model.train()

    ##### Model evaluation ####################################################################################################
    model.eval()
    if save_checkpoints:
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

    return score, output, model.state_dict()


def standard_trainer_reinforce_transfer_core(
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
    baseline_momentum=0.99,
    lr_init_reinforce=0.07,
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

    def full_objective_reinforce(
        model,
        dataloader,
        data_key,
        *args,
        **kwargs,
        ):
        """
        Computes the standard loss (Poisson + regularizers) and REINFORCE loss if available.
        baseline_dict: dict storing moving-average baselines per data_key
        """

        model_output = model(args[0].to(device), data_key=data_key, **kwargs)

        time_left = model_output.shape[1]
        original_data = args[1].transpose(2, 1)[:, -time_left:, :].to(device)
        

        loss_scale = (
            np.sqrt(len(dataloader[data_key].dataset) / args[0].shape[0])
            if scale_loss else 1.0
        )

        if not isinstance(model.core.regularizer(), tuple):
            regularizers = int(not detach_core) * model.core.regularizer() + model.readout.regularizer(data_key)
        else:
            regularizers = int(not detach_core) * sum(model.core.regularizer()) + model.readout.regularizer(data_key)

        loss_per_neuron = criterion(model_output, original_data)
        standard_loss = loss_per_neuron.sum()
        

        reinforce_loss = 0.0

        if hasattr(model.readout[data_key], "last_selected_log_probs"):
            with torch.no_grad():
                baseline_dict[data_key] = (
                    baseline_momentum * baseline_dict[data_key]
                    + (1 - baseline_momentum) * loss_per_neuron
                )

            advantage = loss_per_neuron - baseline_dict[data_key] # [N]
            reinforce_loss = (advantage.detach() * model.readout[data_key].last_selected_log_probs).sum() #[B, N]
            
            total_loss = loss_scale * (standard_loss + reinforce_loss) + regularizers #+ entropy_loss

        return total_loss, standard_loss, reinforce_loss

    ##### Model training ####################################################################################################
    model.to(device)
    set_random_seed(seed)
    model.train()

    criterion = getattr(modules, loss_function)(avg=avg_loss, per_neuron=True)
    stop_closure = partial(
        getattr(scores, stop_function),
        dataloaders=dataloaders[validation_str],
        device=device,
        per_neuron=False,
        avg=True,
        deeplake_ds=deeplake_ds,
    )

    n_iterations = len(LongCycler(dataloaders["train"]))

    if optimizer_type == "Adam":
        optimizer = torch.optim.Adam([p for n, p in model.named_parameters() if "z_logits" in n], lr=lr_init_reinforce)
    elif optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr_init)

    # --- Baselines per readout ---
    baseline_dict = {dk: torch.zeros(model.readout[dk].outdims, device=device)
                 for dk in dataloaders["train"].keys()}
    baseline_momentum = baseline_momentum

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
        epoch_standard_loss = 0
        epoch_reinforce_loss = 0
        epoch_val_loss = 0
        for batch_no, (data_key, data) in tqdm(
            enumerate(LongCycler(dataloaders["train"])),
            total=n_iterations,
            desc="Epoch {}".format(epoch),
        ):
            batch_no_tot += 1
            batch_args = list(data)

            batch_kwargs = data._asdict() if not isinstance(data, dict) else data


            total_loss, standard_loss, reinforce_loss = full_objective_reinforce(
                model,
                dataloaders['train'],
                data_key,
                *batch_args,
                **batch_kwargs
            )

            loss = total_loss / optim_step_count
            loss.backward()

            epoch_loss += loss.detach()
            epoch_standard_loss += standard_loss.detach()
            epoch_reinforce_loss += reinforce_loss.detach()
            if (batch_no + 1) % optim_step_count == 0: # TODO: or (batch_no + 1 == len(LongCycler(dataloaders["train"])))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        model.eval()

        validation_correlation = get_correlations(
            model,
            dataloaders[validation_str],
            device=device,
            as_dict=False,
            per_neuron=False,
            deeplake_ds=deeplake_ds,
        )
        val_loss, val_standard_loss, val_reinforce_loss = full_objective_reinforce(
                model,
                dataloaders[validation_str],
                data_key,
                *batch_args,
                **batch_kwargs,
                detach_core=detach_core,
            )

        print(
            f"Epoch {epoch}, Batch {batch_no}, Train total loss {loss}, Validation loss {val_loss}"
        )
        print(f"EPOCH={epoch}  validation_correlation={validation_correlation}")
        print(
            f"Epoch standard train loss {epoch_standard_loss}, Epoch reinforce train loss {epoch_reinforce_loss}"
        )

        # REINFORCE-specific analysis
        for data_key in dataloaders["train"].keys():
            if hasattr(model.readout[data_key], "z_logits"):
                with torch.no_grad():
                    channel_probs = F.softmax(model.readout[data_key].z_logits, dim=1)
                    entropy = -(channel_probs * torch.log(channel_probs + 1e-8)).sum(dim=1).mean()
                    print(f"  [{data_key}] Channel Selection Entropy: {entropy.item():.4f}")
                    most_popular = channel_probs.argmax(dim=1).bincount(minlength=model.readout[data_key].in_shape[0])
                    top_channels = most_popular.topk(min(10, model.readout[data_key].in_shape[0])).indices.tolist()
                    print(f"  [{data_key}] Most used channels: {top_channels}")



        if use_wandb:
            wandb_dict = {
                "Epoch Train loss": epoch_loss,
                "validation_correlation": validation_correlation,
                "Epoch validation loss": val_loss,
                "Epoch": epoch,
                "entropy": entropy,
                "Epoch train_poisson_loss": epoch_standard_loss,
                "Epoch train_reinforce_loss":epoch_reinforce_loss,
                "val_poisson_loss": val_standard_loss,
                "val_reinforce_loss":val_reinforce_loss,
            }
            wandb.log(wandb_dict)
        model.train()

    ##### Model evaluation ####################################################################################################
    model.eval()
    if save_checkpoints:
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

    return score, output, model.state_dict()

def standard_trainer_reinforce_transfer_z_logits(
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
    baseline_momentum=0.99,
    lr_init_reinforce=0.07,
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

    def full_objective_reinforce(
        model,
        dataloader,
        data_key,
        *args,
        **kwargs,
        ):
        """
        Computes the standard loss (Poisson + regularizers) and REINFORCE loss if available.
        baseline_dict: dict storing moving-average baselines per data_key
        """

        model_output = model(args[0].to(device), data_key=data_key, **kwargs)

        time_left = model_output.shape[1]
        original_data = args[1].transpose(2, 1)[:, -time_left:, :].to(device)
        

        loss_scale = (
            np.sqrt(len(dataloader[data_key].dataset) / args[0].shape[0])
            if scale_loss else 1.0
        )

        if not isinstance(model.core.regularizer(), tuple):
            regularizers = int(not detach_core) * model.core.regularizer() + model.readout.regularizer(data_key)
        else:
            regularizers = int(not detach_core) * sum(model.core.regularizer()) + model.readout.regularizer(data_key)

        loss_per_neuron = criterion(model_output, original_data)
        standard_loss = loss_per_neuron.sum()
        

        reinforce_loss = 0.0

        if hasattr(model.readout[data_key], "last_selected_log_probs"):
            with torch.no_grad():
                baseline_dict[data_key] = (
                    baseline_momentum * baseline_dict[data_key]
                    + (1 - baseline_momentum) * loss_per_neuron
                )

            advantage = loss_per_neuron - baseline_dict[data_key] # [N]
            reinforce_loss = (advantage.detach() * model.readout[data_key].last_selected_log_probs).sum() #[B, N]
            
            total_loss = loss_scale * (standard_loss + reinforce_loss) + regularizers #+ entropy_loss

        return total_loss, standard_loss, reinforce_loss

    ##### Model training ####################################################################################################
    model.to(device)
    set_random_seed(seed)
    model.train()

    criterion = getattr(modules, loss_function)(avg=avg_loss, per_neuron=True)
    stop_closure = partial(
        getattr(scores, stop_function),
        dataloaders=dataloaders[validation_str],
        device=device,
        per_neuron=False,
        avg=True,
        deeplake_ds=deeplake_ds,
    )

    n_iterations = len(LongCycler(dataloaders["train"]))

    if optimizer_type == "Adam":
        optimizer = torch.optim.Adam([   #!!!!! Using the same optimizer only worked when transferring the pre-trained core
                {
                    "params": [p for n, p in model.named_parameters() if "z_logits" in n],  # all z_logits
                    "lr": lr_init_reinforce
                },
                {
                    "params": [p for n, p in model.named_parameters() if "z_logits" not in n],  # everything else
                    "lr": lr_init
                }
            ])
    elif optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr_init)

    # --- Baselines per readout ---
    baseline_dict = {dk: torch.zeros(model.readout[dk].outdims, device=device)
                 for dk in dataloaders["train"].keys()}
    baseline_momentum = baseline_momentum

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
        epoch_standard_loss = 0
        epoch_reinforce_loss = 0
        epoch_val_loss = 0
        for batch_no, (data_key, data) in tqdm(
            enumerate(LongCycler(dataloaders["train"])),
            total=n_iterations,
            desc="Epoch {}".format(epoch),
        ):
            batch_no_tot += 1
            batch_args = list(data)

            batch_kwargs = data._asdict() if not isinstance(data, dict) else data


            total_loss, standard_loss, reinforce_loss = full_objective_reinforce(
                model,
                dataloaders['train'],
                data_key,
                *batch_args,
                **batch_kwargs
            )

            loss = total_loss / optim_step_count
            loss.backward()

            epoch_loss += loss.detach()
            epoch_standard_loss += standard_loss.detach()
            epoch_reinforce_loss += reinforce_loss.detach()
            if (batch_no + 1) % optim_step_count == 0: # TODO: or (batch_no + 1 == len(LongCycler(dataloaders["train"])))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        model.eval()

        validation_correlation = get_correlations(
            model,
            dataloaders[validation_str],
            device=device,
            as_dict=False,
            per_neuron=False,
            deeplake_ds=deeplake_ds,
        )
        val_loss, val_standard_loss, val_reinforce_loss = full_objective_reinforce(
                model,
                dataloaders[validation_str],
                data_key,
                *batch_args,
                **batch_kwargs,
                detach_core=detach_core,
            )

        print(
            f"Epoch {epoch}, Batch {batch_no}, Train total loss {epoch_loss}, Validation loss {val_loss}"
        )
        print(f"EPOCH={epoch}  validation_correlation={validation_correlation}")
        print(
            f"Epoch standard train loss {epoch_standard_loss}, Epoch reinforce train loss {epoch_reinforce_loss}"
        )

        # REINFORCE-specific analysis
        for data_key in dataloaders["train"].keys():
            if hasattr(model.readout[data_key], "z_logits"):
                with torch.no_grad():
                    channel_probs = F.softmax(model.readout[data_key].z_logits, dim=1)
                    entropy = -(channel_probs * torch.log(channel_probs + 1e-8)).sum(dim=1).mean()
                    print(f"  [{data_key}] Channel Selection Entropy: {entropy.item():.4f}")
                    most_popular = channel_probs.argmax(dim=1).bincount(minlength=model.readout[data_key].in_shape[0])
                    top_channels = most_popular.topk(min(10, model.readout[data_key].in_shape[0])).indices.tolist()
                    print(f"  [{data_key}] Most used channels: {top_channels}")



        if use_wandb:
            wandb_dict = {
                "Epoch Train loss": epoch_loss,
                "validation_correlation": validation_correlation,
                "Epoch validation loss": val_loss,
                "Epoch": epoch,
                "entropy": entropy,
                "Epoch train_poisson_loss": epoch_standard_loss,
                "Epoch train_reinforce_loss":epoch_reinforce_loss,
                "val_poisson_loss": val_standard_loss,
                "val_reinforce_loss":val_reinforce_loss,
                "Learning rate z_logits": optimizer.param_groups[0]['lr'],
                "Learning rate rest": optimizer.param_groups[1]['lr'],
            }
            wandb.log(wandb_dict)
        model.train()

    ##### Model evaluation ####################################################################################################
    model.eval()
    if save_checkpoints:
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

    return score, output, model.state_dict()

def standard_trainer_reinforce_max_entropy(
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
    baseline_momentum=0.99,
    lr_init_reinforce=0.07,
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

    def full_objective_reinforce(
        model,
        dataloader,
        data_key,
        *args,
        **kwargs,
        ):
        """
        Computes the standard loss (Poisson + regularizers) and REINFORCE loss if available.
        baseline_dict: dict storing moving-average baselines per data_key
        """

        model_output = model(args[0].to(device), data_key=data_key, **kwargs)

        time_left = model_output.shape[1]
        original_data = args[1].transpose(2, 1)[:, -time_left:, :].to(device)
        

        loss_scale = (
            np.sqrt(len(dataloader[data_key].dataset) / args[0].shape[0])
            if scale_loss else 1.0
        )

        if not isinstance(model.core.regularizer(), tuple):
            regularizers = int(not detach_core) * model.core.regularizer() + model.readout.regularizer(data_key)
        else:
            regularizers = int(not detach_core) * sum(model.core.regularizer()) + model.readout.regularizer(data_key)

        loss_per_neuron = criterion(model_output, original_data)
        standard_loss = loss_per_neuron.sum()
        

        reinforce_loss = 0.0

        if hasattr(model.readout[data_key], "last_selected_log_probs"):
            with torch.no_grad():
                baseline_dict[data_key] = (
                    baseline_momentum * baseline_dict[data_key]
                    + (1 - baseline_momentum) * loss_per_neuron
                )

            advantage = loss_per_neuron - baseline_dict[data_key] # [N]
            reinforce_loss = (advantage.detach() * model.readout[data_key].last_selected_log_probs).sum() #[B, N]
        
        # Entropy loss
        if epoch < 40:
            entropy_factor = -1.0   # maximize entropy (encourage exploration)
        else:
            entropy_factor = 0.0
        
        if hasattr(model.readout[data_key], "z_logits"):
            z_probs_for_entropy = F.softmax(model.readout[data_key].z_logits, dim=1)  
            entropy_per_neuron = -(z_probs_for_entropy * z_probs_for_entropy.log()).sum(dim=1)  
            
            entropy_sum = entropy_per_neuron.sum()

            # --- dynamic beta_entropy with EMA ---
            with torch.no_grad():
                current_mag = reinforce_loss.abs().item()
                reinforce_mag_ema[data_key] = (
                    ema_momentum * reinforce_mag_ema[data_key]
                    + (1 - ema_momentum) * current_mag
                )
                denom = entropy_sum.abs().item() + 1e-8
                beta_entropy_dynamic = reinforce_mag_ema[data_key] / denom

            entropy_loss = beta_entropy_dynamic * entropy_factor * entropy_sum
        else:
            entropy_loss = 0.0
            
        total_loss = loss_scale * (standard_loss + reinforce_loss + entropy_loss) + regularizers #+ entropy_loss

        return total_loss, standard_loss, reinforce_loss, entropy_loss

    ##### Model training ####################################################################################################
    model.to(device)
    set_random_seed(seed)
    model.train()

    criterion = getattr(modules, loss_function)(avg=avg_loss, per_neuron=True)
    stop_closure = partial(
        getattr(scores, stop_function),
        dataloaders=dataloaders[validation_str],
        device=device,
        per_neuron=False,
        avg=True,
        deeplake_ds=deeplake_ds,
    )

    n_iterations = len(LongCycler(dataloaders["train"]))

    if optimizer_type == "Adam":
        # optimizer = torch.optim.Adam([   #!!!!! Using the same optimizer only worked when transferring the pre-trained core
        #         {
        #             "params": [p for n, p in model.named_parameters() if "z_logits" in n],  # all z_logits
        #             "lr": lr_init_reinforce
        #         },
        #         {
        #             "params": [p for n, p in model.named_parameters() if "z_logits" not in n],  # everything else
        #             "lr": lr_init
        #         }
        #     ])
        optimizer = torch.optim.Adam(
            [p for n, p in model.named_parameters() if "z_logits" not in n], lr=lr_init
        )
        optimizer_z = torch.optim.Adam(
            [p for n, p in model.named_parameters() if "z_logits" in n], lr=lr_init_reinforce
        )
    elif optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr_init)

    # --- Baselines per readout ---
    baseline_dict = {dk: torch.zeros(model.readout[dk].outdims, device=device)
                 for dk in dataloaders["train"].keys()}
    baseline_momentum = baseline_momentum

    reinforce_mag_ema = {key: 0.0 for key in dataloaders["train"].keys()}
    ema_momentum = 0.9  # decay factor, adjust as needed

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
        optimizer_z.zero_grad(set_to_none=True)
        epoch_loss = 0
        epoch_standard_loss = 0
        epoch_reinforce_loss = 0
        epoch_entropy_loss = 0
        epoch_val_loss = 0
        for batch_no, (data_key, data) in tqdm(
            enumerate(LongCycler(dataloaders["train"])),
            total=n_iterations,
            desc="Epoch {}".format(epoch),
        ):
            batch_no_tot += 1
            batch_args = list(data)

            batch_kwargs = data._asdict() if not isinstance(data, dict) else data


            total_loss, standard_loss, reinforce_loss, entropy_loss = full_objective_reinforce(
                model,
                dataloaders['train'],
                data_key,
                *batch_args,
                **batch_kwargs
            )

            loss = total_loss / optim_step_count
            loss.backward()

            epoch_loss += loss.detach()
            epoch_standard_loss += standard_loss.detach()
            epoch_reinforce_loss += reinforce_loss.detach()
            epoch_entropy_loss += entropy_loss.detach()
            if (batch_no + 1) % optim_step_count == 0: # TODO: or (batch_no + 1 == len(LongCycler(dataloaders["train"])))
                optimizer.step()
                optimizer_z.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_z.zero_grad(set_to_none=True)
                

        model.eval()

        validation_correlation = get_correlations(
            model,
            dataloaders[validation_str],
            device=device,
            as_dict=False,
            per_neuron=False,
            deeplake_ds=deeplake_ds,
        )
        val_loss, val_standard_loss, val_reinforce_loss, val_entropy_loss = full_objective_reinforce(
                model,
                dataloaders[validation_str],
                data_key,
                *batch_args,
                **batch_kwargs,
                detach_core=detach_core,
            )

        print(
            f"Epoch {epoch}, Batch {batch_no}, Train total loss {epoch_loss}, Validation loss {val_loss}"
        )
        print(f"EPOCH={epoch}  validation_correlation={validation_correlation}")
        print(
            f"Epoch standard train loss {epoch_standard_loss}, Epoch reinforce train loss {epoch_reinforce_loss}"
        )

        # REINFORCE-specific analysis
        for data_key in dataloaders["train"].keys():
            if hasattr(model.readout[data_key], "z_logits"):
                with torch.no_grad():
                    channel_probs = F.softmax(model.readout[data_key].z_logits, dim=1)
                    entropy = -(channel_probs * torch.log(channel_probs + 1e-8)).sum(dim=1).mean()
                    print(f"  [{data_key}] Channel Selection Entropy: {entropy.item():.4f}")
                    most_popular = channel_probs.argmax(dim=1).bincount(minlength=model.readout[data_key].in_shape[0])
                    top_channels = most_popular.topk(min(10, model.readout[data_key].in_shape[0])).indices.tolist()
                    print(f"  [{data_key}] Most used channels: {top_channels}")



        if use_wandb:
            wandb_dict = {
                "Epoch Train loss": epoch_loss,
                "validation_correlation": validation_correlation,
                "Epoch validation loss": val_loss,
                "Epoch": epoch,
                "entropy": entropy,
                "Epoch train_poisson_loss": epoch_standard_loss,
                "Epoch train_reinforce_loss":epoch_reinforce_loss,
                "Epoch train_entropy_loss":epoch_entropy_loss,
                "val_poisson_loss": val_standard_loss,
                "val_reinforce_loss":val_reinforce_loss,
                "Learning rate z_logits": optimizer.param_groups[0]['lr'],
                "Learning rate rest": optimizer_z.param_groups[0]['lr'],
            }
            wandb.log(wandb_dict)
        model.train()

    ##### Model evaluation ####################################################################################################
    model.eval()
    if save_checkpoints:
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

    return score, output, model.state_dict()