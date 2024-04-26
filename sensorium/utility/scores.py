import warnings

import numpy as np
import torch
from neuralpredictors.measures.np_functions import corr
from neuralpredictors.training import device_state
from nnfabrik.builder import get_data
import operator


def model_predictions(
    model, dataloader, data_key, device="cpu", skip=50, deeplake_ds=False,
    pad_responses=False,
):
    """
    computes model predictions for a given dataloader and a model
    Returns:
        target: ground truth, i.e. neuronal firing rates of the neurons
        output: responses as predicted by the network, a list of arrays, each array
                corresponds to responses of one trial, array shape: (num_of_neurons, num_of_frames_per_trial)
    """

    target, output = [], []
    for batch in dataloader:
        batch_kwargs = batch._asdict() if not isinstance(batch, dict) else batch
        if deeplake_ds:
            for k in batch_kwargs.keys():
                if k not in ["id", "index"]:
                    batch_kwargs[k] = torch.Tensor(np.asarray(batch_kwargs[k])).to(
                        device
                    )
            images = batch_kwargs["videos"]
            responses = batch_kwargs["responses"]
        else:
            images, responses = (
                batch[:2]
                if not isinstance(batch, dict)
                else (batch["videos"], batch["responses"])
            )
        if pad_responses:
            # images shape: [b, c, t, h, w]
            temp_mean_image_along_t = torch.mean(images, dim=2, keepdim=True).repeat(1, 1, 50, 1, 1)
            images = torch.cat((temp_mean_image_along_t, images), dim=2)
            
            # responses shape: [b, n, t]    `n` means neurons
            responses = torch.nn.functional.pad(responses, (skip,0), value=1e-8, mode='constant')

            # batch_kwargs['pupil_center'], with shape [b, c, t], is also passed as parameter in the model() inference
            # this means that it has to be padded as well
            batch_kwargs['pupil_center'] = torch.nn.functional.pad(batch_kwargs['pupil_center'], (skip,0), value=1e-8, mode='constant')

        with torch.no_grad():
            resp = responses.detach().cpu().numpy()[:, :, skip:]
            target = target + list(resp)
            with device_state(model, device):
                out = (
                    model(images.to(device), data_key=data_key, **batch_kwargs)
                    .detach()
                    .cpu()[:, -resp.shape[-1] :, :]
                )
                assert (
                    out.shape[1] == resp.shape[-1]
                ), f"model prediction is too short ({out.shape[1]} vs {resp.shape[-1]})"
                output = output + list(out.permute(0, 2, 1).numpy())

    return target, output


def get_correlations(
    model,
    dataloaders,
    tier=None,
    device="cpu",
    as_dict=False,
    per_neuron=True,
    deeplake_ds=False,
    **kwargs,
):
    """
    Computes single-trial correlation between model prediction and true responses
    Args:
        model (torch.nn.Module): Model used to predict responses.
        dataloaders (dict): dict of test set torch dataloaders.
        tier(str): the data-tier (train/test/val). If tier is None, then it is assumed that the the tier-key is not present.
        device (str, optional): device to compute on. Defaults to "cpu".
        as_dict (bool, optional): whether to return the results per data_key. Defaults to False.
        per_neuron (bool, optional): whether to return the results per neuron or averaged across neurons. Defaults to True.
    Returns:
        dict or np.ndarray: contains the correlation values.
    """
    correlations = {}
    dl = dataloaders[tier] if tier is not None else dataloaders
    for k, v in dl.items():
        target, output = model_predictions(
            dataloader=v,
            model=model,
            data_key=k,
            device=device,
            deeplake_ds=deeplake_ds,
        )
        target = np.concatenate(
            target, axis=1
        ).T  # , shape: (num_of_frames_all_trials, num_of_neurons)
        output = np.concatenate(output, axis=1).T
        correlations[k] = corr(target, output, axis=0)

        if np.any(np.isnan(correlations[k])):
            warnings.warn(
                "{}% NaNs , NaNs will be set to Zero.".format(
                    np.isnan(correlations[k]).mean() * 100
                )
            )
        correlations[k][np.isnan(correlations[k])] = 0

    if not as_dict:
        correlations = (
            np.hstack([v for v in correlations.values()])
            if per_neuron
            else np.mean(np.hstack([v for v in correlations.values()]))
        )
    return correlations


def get_poisson_loss(
    model,
    dataloaders,
    device="cpu",
    as_dict=False,
    avg=False,
    per_neuron=True,
    eps=1e-12,
):
    poisson_loss = {}
    for k, v in dataloaders.items():
        target, output = model_predictions(
            dataloader=v, model=model, data_key=k, device=device
        )
        loss = output - target * np.log(output + eps)
        poisson_loss[k] = np.mean(loss, axis=0) if avg else np.sum(loss, axis=0)
    if as_dict:
        return poisson_loss
    else:
        if per_neuron:
            return np.hstack([v for v in poisson_loss.values()])
        else:
            return (
                np.mean(np.hstack([v for v in poisson_loss.values()]))
                if avg
                else np.sum(np.hstack([v for v in poisson_loss.values()]))
            )


def get_data_filetree_loader(
    filename=None, dataloader=None, tier="test", stimulus_type=None
):
    """
    Extracts necessary data for model evaluation from a dataloader based on the FileTree dataset.

    Args:
        filename (str): Specifies a path to the FileTree dataset.
        dataloader (obj): PyTorch Dataloader
        stimulus_type (str): such as "clip" and "dotsequence"

    Returns:
        tuple: Contains:
               - tier_hashes (1D array, the length is equal to the number of trial in that tier)
               - evaluation_hashes_unique (1D array, unique condition_hash for evaluation)

    """

    if dataloader is None:
        dataset_fn = "sensorium.datasets.mouse_video_loaders.mouse_video_loader"
        dataset_config = {
            "paths": filename,
            "normalize": True,
            "tier": tier,
            "batch_size": 1,
            "frames": 300,
            "to_cut": False,
        }
        dataloaders = get_data(dataset_fn, dataset_config)
        data_key = list(dataloaders[tier].keys())[0]

        dat = dataloaders[tier][data_key].dataset
    else:
        dat = dataloader.dataset

    # neuron_ids = dat.neurons.unit_ids.tolist()
    tiers = dat.trial_info.tiers
    complete_condition_hashes = dat.trial_info.condition_hash
    # complete_trial_idx = dat.trial_info.trial_idx

    tier_hashes = complete_condition_hashes[
        np.where(tiers == tier)[0]
    ]  # condition_hashes for the specific tier
    if (
        stimulus_type != None
    ):  # condition_hashes for the specific tier and the specific stimulus_type
        complete_trial_ts = getattr(dat.trial_info, stimulus_type + "_trial_ts")
        tier_stimlus_type_hashes = complete_condition_hashes[
            np.where((tiers == tier) & (complete_trial_ts != "NaT"))[0]
        ]

    # the condition_hash that would be used for evaluation
    evaluation_hashes_unique = (
        np.unique(tier_stimlus_type_hashes)
        if stimulus_type != None
        else np.unique(tier_hashes)
    )

    return tier_hashes, evaluation_hashes_unique


def get_signal_correlations(
    model,
    dataloaders,
    tier,
    stimulus_type=None,
    evaluation_hashes_unique=None,
    device="cpu",
    as_dict=False,
    per_neuron=True,
    pad_responses=False,
):
    """
    Similar as `get_correlations` but first responses and predictions are averaged across repeats
    and then the correlation is computed. In other words, the correlation is computed between
    the means across repeats.
    For the test loaders, we may have different stimulus_types, such as clip and dotsequence,
    we may compute the correlations for some specific stimulus_type.

    Args:
        dataloaders (obj): PyTorch Dataloaders, without tier
        tier (str):
        stimulus_type (str): such as "clip" and "dotsequence"
        evaluation_hashes_unique: 1D array, unique condition_hash for evaluation

    Returns:
        evaluation_hashes_unique
        single_trial_corrs
        mean_corrs
    """
    mean_corrs = {}
    single_trial_corrs = {}
    for data_key, dataloader in dataloaders[tier].items():
        tier_hashes, evaluation_hashes_unique_temp = get_data_filetree_loader(
            dataloader=dataloader, tier=tier, stimulus_type=stimulus_type
        )
        if evaluation_hashes_unique is None:
            evaluation_hashes_unique = evaluation_hashes_unique_temp

        responses, predictions = model_predictions(
            model, dataloader, data_key=data_key, device=device, pad_responses=pad_responses
        )
        responses_align = [
            operator.itemgetter(*(np.where(tier_hashes == temp)[0]))(responses)
            for temp in evaluation_hashes_unique
        ]
        predictions_align = [
            operator.itemgetter(*(np.where(tier_hashes == temp)[0]))(predictions)
            for temp in evaluation_hashes_unique
        ]
        
        # in some cases, each repeat may be presented with distinct time frames (1_frame or 2_frame difference)
        for num in range(len(responses_align)):
            frames_per_repeat = np.array([ii.shape[1] for ii in responses_align[num]])
            if len(np.unique(frames_per_repeat))>1: # number of time frames for each repeat are different
                print (f'Warning: responses_align[{num}] have multiple time frames for repeats: {frames_per_repeat}')
                responses_align[num]   = [ii[:,-np.min(frames_per_repeat):] for ii in responses_align[num]]
                predictions_align[num] = [ii[:,-np.min(frames_per_repeat):] for ii in predictions_align[num]]

        responses_align = [
            np.transpose(np.array(temp), (0, 2, 1)) for temp in responses_align
        ]
        # responses_align: a list of array, each array corresponds to reponses to one condition_hash,
        # array shape (num_of_repeats_for_that_hash, num_of_frames_for_that_trial, num_of_neurons)
        predictions_align = [
            np.transpose(np.array(temp), (0, 2, 1)) for temp in predictions_align
        ]

        # mean correlations
        temp_responses_align = np.concatenate(
            [np.mean(temp, axis=0) for temp in responses_align], axis=0
        )
        temp_predictions_align = np.concatenate(
            [np.mean(temp, axis=0) for temp in predictions_align], axis=0
        )
        mean_corrs[data_key] = corr(
            temp_responses_align, temp_predictions_align, axis=0
        )
        # single trial correlations
        temp_responses_align = np.concatenate(responses_align, axis=0)
        temp_predictions_align = np.concatenate(predictions_align, axis=0)
        temp_responses_align = np.reshape(
            temp_responses_align, (-1, temp_responses_align.shape[-1])
        )
        temp_predictions_align = np.reshape(
            temp_predictions_align, (-1, temp_predictions_align.shape[-1])
        )
        single_trial_corrs[data_key] = corr(
            temp_responses_align, temp_predictions_align, axis=0
        )
        del temp_responses_align, temp_predictions_align

    if not as_dict:
        mean_corrs = (
            np.hstack([v for v in mean_corrs.values()])
            if per_neuron
            else np.mean(np.hstack([v for v in mean_corrs.values()]))
        )
        single_trial_corrs = (
            np.hstack([v for v in single_trial_corrs.values()])
            if per_neuron
            else np.mean(np.hstack([v for v in single_trial_corrs.values()]))
        )
    elif not per_neuron:
        mean_corrs_ = {}
        single_trial_corrs_ = {}
        for k in mean_corrs.keys():
            mean_corrs_[k] = (
                np.hstack([v for v in mean_corrs[k]])
                if per_neuron
                else np.mean(np.hstack([v for v in mean_corrs[k]]))
            )
            single_trial_corrs_[k] = (
                np.hstack([v for v in single_trial_corrs[k]])
                if per_neuron
                else np.mean(np.hstack([v for v in single_trial_corrs[k]]))
            )
        single_trial_corrs = single_trial_corrs_
        mean_corrs = mean_corrs_

    return evaluation_hashes_unique, single_trial_corrs, mean_corrs


def model_predictions_align(
    model,
    dataloaders,
    tier,
    stimulus_type=None,
    evaluation_hashes_unique=None,
    device="cpu",
):
    """
    Similar as `model_predictions` but recorded/predicted responses to repeats are arranged
    by condition_hash, this alignment is for computing mean correlation.
    For the test loaders, we may have different stimulus_types, such as clip and dotsequence,
    we may compute the correlations for some specific stimulus_type.

    Args:
        dataloaders (obj): PyTorch Dataloaders, without tier
        tier (str):
        stimulus_type (str): such as "clip" and "dotsequence"
        evaluation_hashes_unique: 1D array, unique condition_hash for evaluation

    Returns:
        responses_align
        predictions_align
    """
    responses_aligns = {}
    predictions_aligns = {}
    for data_key, dataloader in dataloaders[tier].items():
        tier_hashes, evaluation_hashes_unique_temp = get_data_filetree_loader(
            dataloader=dataloader, tier=tier, stimulus_type=stimulus_type
        )
        if evaluation_hashes_unique is None:
            evaluation_hashes_unique = evaluation_hashes_unique_temp

        responses, predictions = model_predictions(
            model, dataloader, data_key=data_key, device=device
        )
        responses_align = [
            operator.itemgetter(*(np.where(tier_hashes == temp)[0]))(responses)
            for temp in evaluation_hashes_unique
        ]
        predictions_align = [
            operator.itemgetter(*(np.where(tier_hashes == temp)[0]))(predictions)
            for temp in evaluation_hashes_unique
        ]

        # in some cases, each repeat may be presented with distinct time frames (1_frame or 2_frame difference)
        for num in range(len(responses_align)):
            frames_per_repeat = np.array([ii.shape[1] for ii in responses_align[num]])
            if len(np.unique(frames_per_repeat))>1: # number of time frames for each repeat are different
                print (f'Warning: responses_align[{num}] have multiple time frames for repeats: {frames_per_repeat}')
                responses_align[num]   = [ii[:,-np.min(frames_per_repeat):] for ii in responses_align[num]]
                predictions_align[num] = [ii[:,-np.min(frames_per_repeat):] for ii in predictions_align[num]]

        responses_align = [
            np.transpose(np.array(temp), (0, 2, 1)) for temp in responses_align
        ]
        # responses_align: a list of array, each array corresponds to reponses to one condition_hash,
        # array shape (num_of_repeats_for_that_hash, num_of_frames_for_that_trial, num_of_neurons)
        predictions_align = [
            np.transpose(np.array(temp), (0, 2, 1)) for temp in predictions_align
        ]
        responses_aligns[data_key] = responses_align
        predictions_aligns[data_key] = predictions_align
        
    return evaluation_hashes_unique, responses_aligns, predictions_aligns
