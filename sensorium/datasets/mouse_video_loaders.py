import numpy as np
from neuralpredictors.data.datasets import (
    MovieFileTreeDataset,
    NRandomSubSequenceDataset,
)
from neuralpredictors.data.samplers import SubsetSequentialSampler
from neuralpredictors.data.transforms import (
    AddBehaviorAsChannels,
    AddPupilCenterAsChannels,
    ChangeChannelsOrder,
    CutVideos,
    ExpandChannels,
    NeuroNormalizer,
    ScaleInputs,
    SelectInputChannel,
    Subsample,
    Subsequence,
    ToTensor,
    SelectBehaviorChannels,
)
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from nnfabrik.utility.nn_helpers import set_random_seed


def mouse_video_loader(
    paths,
    batch_size,
    seed: int = None,
    neuron_ids: np.array = None,
    normalize=True,
    exclude: str = None,
    cuda: bool = False,
    max_frame=None,
    frames=300,
    offset=-1,
    inputs_mean=None,
    inputs_std=None,
    subtract_response_min=False,
    include_behavior=True,
    include_pupil_centers=True,
    include_pupil_centers_as_channels=False,
    scale=1,
    to_cut=False,
    behavior_channels=[0, 1],
    random_sample_within_snippet_flag=False,
    num_random_subsequence=10,
    subsequence_length=100,
    sequence_length=300,
    random_start=None,
    num_workers=4,
    # pin_memory: allocates CPU tensors in page-locked memory so the DataLoader→GPU
    # transfer is done via async DMA (faster). Safe to use even when cuda=False in
    # ToTensor — pinning happens in the main process, not in workers.
    # Set to False for CPU-only inference to avoid wasting pinned RAM.
    pin_memory=True,
):
    """
    Symplified version of the sensorium mouse_loaders.py
     Returns a dictionary of dataloaders (i.e., trainloaders, valloaders, and testloaders) for >= 1 dataset(s).
    Args:
        paths (list): list of paths for the datasets
        batch_size (int): batch size.
        seed (int): seed. Not really needed. But nnFabrik requires it.
        frames (int, optional): how many frames ot take per video
        max_frame (int, optional): which is the maximal frame that could be taken per video
        offset (int, optional): Offset to start the subsequence from. Defaults to -1, corresponding to random but valid offset at each iteration.
        subtract_response_min (bool, optional): whether to subtract response minimum (for fluorescence data). Defulats to 'False'.
        cuda (bool, optional): whether to place the data on gpu or not.
        normalize (bool, optional): whether to normalize the data (see also exclude)
        exclude (str, optional): data to exclude from data-normalization. Only relevant if normalize=True. Defaults to 'images'
        include_behavior (bool, optional): whether to include behavioral data
        include_pupil_centers (bool, optional): whether to include pupil center data
        include_pupil_centers_as_channels(bool, optional): whether to include pupil center data as channels
        scale(float, optional): scalar factor for the image resolution.
            scale = 1: full iamge resolution (144 x 256)
            scale = 0.25: resolution used for model training (36 x 64)
        float64 (bool, optional):  whether to use float64 in MovieFileTreeDataset
        behavior_channels: behavior data has a shape (number_of_channels, number_of_frames)
            For sensorium2023, we have two channels (pupil size, locomotion), for example, the shape of behavior data could be (2,300).
                So the default behavior_channels=[0, 1] works for sensorium2023.
            For the nexport output, we have three channels (pupil size, pupil size change, locomotion).
                For now, the first two channels have the same content, i.e., pupil size. We use behavior_channels=[0,2] for training.
        random_sample_within_snippet_flag: whether to use random sampling for each training snippet or not.
            Default: False, so no random sampling.
        num_random_subsequence: if we set random_sample_within_snippet_flag=True, this specifies the number of random subsequence
            within each snippet.
        subsequence_length: if we set random_sample_within_snippet_flag=True, this specifies the length of each subsequence.
        sequence_length: number of frames in each training snippet, usually it is 300 or 299.
        random_start: if we set random_sample_within_snippet_flag=True, this specifies the start points of each subsequence.
    Returns:
        dict: dictionary of dictionaries where the first level keys are 'train', 'validation', and 'test', and second level keys are data_keys.
    """

    if seed is not None:
        set_random_seed(seed)
    data_keys = [
        "videos",
        "responses",
    ]
    if include_behavior:
        data_keys.append("behavior")
    if include_pupil_centers:
        data_keys.append("pupil_center")

    #     dataloaders_combined = {"validation": {}, "train": {}, "test": {}}
    dataloaders_combined = {}

    for path_id, path in enumerate(paths):
        dat2 = MovieFileTreeDataset(path, *data_keys)

        if neuron_ids is None:
            conds = np.ones(len(dat2.neurons.cell_motor_coordinates), dtype=bool)
            idx = np.where(conds)[0]
        else:
            idx = np.copy(neuron_ids[path_id])

        more_transforms = [
            Subsample(idx, target_index=0),
            CutVideos(
                max_frame=max_frame,
                frame_axis={data_key: -1 for data_key in data_keys},
                target_groups=data_keys,
            ),
            ChangeChannelsOrder((2, 0, 1), in_name="videos"),
            ChangeChannelsOrder((1, 0), in_name="responses"),
        ]
        if include_behavior:
            more_transforms.append(SelectBehaviorChannels(channels=behavior_channels))
            more_transforms.append(ChangeChannelsOrder((1, 0), in_name="behavior"))
        if include_pupil_centers:
            more_transforms.append(ChangeChannelsOrder((1, 0), in_name="pupil_center"))

        if to_cut:
            more_transforms.append(
                Subsequence(frames=frames, channel_first=(), offset=offset)
            )
        more_transforms = more_transforms + [
            ChangeChannelsOrder((1, 0), in_name="responses"),
            ExpandChannels("videos"),
        ]
        if include_behavior:
            more_transforms.append(ChangeChannelsOrder((1, 0), in_name="behavior"))
        if include_pupil_centers:
            more_transforms.append(ChangeChannelsOrder((1, 0), in_name="pupil_center"))

        if include_behavior:
            more_transforms.append(AddBehaviorAsChannels("videos"))
        if include_pupil_centers and include_pupil_centers_as_channels:
            more_transforms.append(AddPupilCenterAsChannels("videos"))

        more_transforms.append(ToTensor(cuda))
        more_transforms.insert(
            0, ScaleInputs(scale=scale, in_name="videos", channel_axis=-1)
        )
        if normalize:
            try:
                more_transforms.insert(
                    0,
                    NeuroNormalizer(
                        dat2,
                        exclude=exclude,
                        inputs_mean=inputs_mean,
                        inputs_std=inputs_std,
                        subtract_response_min=subtract_response_min,
                        in_name="videos",
                    ),
                )
            except:
                more_transforms.insert(
                    0,
                    NeuroNormalizer(
                        dat2,
                        exclude=exclude,
                        subtract_response_min=subtract_response_min,
                        in_name="videos",
                    ),
                )

        dat2.transforms.extend(more_transforms)

        if random_sample_within_snippet_flag == False:
            # subsample images
            tier = None
            dataloaders = {}
            keys = [tier] if tier else list(set(list(dat2.trial_info.tiers)))
            tier_array = dat2.trial_info.tiers

            for tier in keys:
                if tier != "none":
                    subset_idx = np.where(tier_array == tier)[0]

                    sampler = (
                        SubsetRandomSampler(subset_idx)
                        if tier == "train"
                        else SubsetSequentialSampler(subset_idx)
                    )
                    dataloaders[tier] = DataLoader(
                        dat2,
                        sampler=sampler,
                        batch_size=batch_size if tier == "train" else 1,
                        num_workers=num_workers,
                        pin_memory=pin_memory,
                        persistent_workers=num_workers > 0,
                    )

        else:  # perform the random sampling within each snippet
            dat3 = NRandomSubSequenceDataset(
                original_dat=dat2,
                num_random_subsequence=num_random_subsequence,
                subsequence_length=subsequence_length,
                sequence_length=sequence_length,
                random_start=random_start,
            )

            tier = None
            dataloaders = {}
            keys = [tier] if tier else list(set(list(dat2.trial_info.tiers)))
            for tier in keys:
                if tier != "none":
                    subset_idx = np.where(np.array(dat3.new_tiers) == tier)[0]
                    sampler = (
                        SubsetRandomSampler(subset_idx)
                        if tier == "train"
                        else SubsetSequentialSampler(subset_idx)
                    )
                    dataloaders[tier] = DataLoader(
                        dat3,
                        sampler=sampler,
                        batch_size=batch_size if tier == "train" else 1,
                        num_workers=num_workers,
                        pin_memory=pin_memory,
                        persistent_workers=num_workers > 0,
                    )

        dataset_name = path.split("/")[-2]
        for k, v in dataloaders.items():
            if k not in dataloaders_combined.keys():
                dataloaders_combined[k] = {}
            dataloaders_combined[k][dataset_name] = v

    return dataloaders_combined
