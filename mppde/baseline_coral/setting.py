import einops

from torch.utils.data import Dataset


def init_setting(dataset, setting, sequence_length_in, sequence_length_out):
    sequence_length_total = sequence_length_in + sequence_length_out
    if setting == 'all':
        return setting_all(dataset)
    elif setting == 'extrapolation':
        return setting_ext(dataset, sequence_length_in, sequence_length_out)
    elif setting == 'interpolation':
        return setting_int(dataset, sequence_length_total, sequence_length_in, sequence_length_out)
    else:
        raise NotImplementedError(
            "Setting not implemented. Use 'all' or 'extrapolaiton'")


def setting_all(dataset):
    """
    Basic setting where all frame are used for training

    Args:
        dataset (dataset): input dataset with format N, T, Dx, Dy, d

    Returns:
        datasets: dataset with format N, T, Dx, Dy, d
    """

    return dataset, None, None


def setting_ext(dataset, seen_in=20, seen_out=20):
    """
    Setting for extrapolation where the network learns on seen frame, and extrapolates on the remaining
    Just cut the trajectories in in/out

    Args:
        dataset (dataset): input dataset with format N, T, Dx, Dy, d
        seen (int): length of sequence 

    Returns:
        datasets: dataset with format N/seen, T, Dx, Dy, d
    """

    dataset_in = dataset[:, :seen_in, ...]
    dataset_out = dataset[:, seen_in:seen_in+seen_out, ...]
    dataset_extrapolation = dataset[:, :seen_in+seen_out, ...]

    return dataset_in, dataset_out, dataset_extrapolation


def setting_int(dataset, sequence_length_tot, seen_in=10, seen_out=10):
    """
    Setting w/ "interpolation" as in Dino
    1/ cut trajectories using sequence_length
    2/ cut time for in/out datasets

    Args:
        dataset (dataset): input dataset with format N, T, Dx, Dy, d
        seen (int): length of sequence

    Returns:
        datasets: dataset with format N, T, Dx, Dy, d
    """
    # start from a big trajectory of size 40 for example and suppose we want to cut it in 10-10/10-10
    # reshape trajectories in 2 sub trajectories of size 20
    dataset = einops.rearrange(
        dataset, "b (d t) w h -> (b d) t w h", t=sequence_length_tot
    )

    # cut the time to extract in/out data
    dataset_in = dataset[:, :seen_in, ...]
    dataset_out = dataset[:, seen_in:seen_in + seen_out, ...]
    dataset_extrapolation = dataset[:, :seen_in+seen_out, ...]

    return dataset_in, dataset_out, dataset_extrapolation
