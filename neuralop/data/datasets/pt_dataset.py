from functools import partialmethod
from pathlib import Path
from typing import List, Union, Optional

import torch

from .tensor_dataset import TensorDataset
from ..transforms.data_processors import DefaultDataProcessor
from ..transforms.normalizers import UnitGaussianNormalizer


class PTDataset:
    """PTDataset is a base Dataset class for our library.
        PTDatasets contain input-output pairs a(x), u(x) and may also
        contain additional information, e.g. function parameters,
        input geometry or output query points.

        datasets may implement a download flag at init, which provides
        access to a number of premade datasets for sample problems provided
        in our Zenodo archive.

    Parameters
    ----------
    root_dir : Union[Path, str]
        root at which to download data files
    dataset_name : str
        prefix of pt data files to store/access
    n_train : int
        number of train instances
    n_tests : List[int]
        number of test instances per test dataset
    batch_size : int
        batch size of training set
    test_batch_sizes : List[int]
        batch size of test sets
    train_resolution : int
        resolution of data for training set
    test_resolutions : List[int]
        resolution of data for testing sets
    encode_input : bool, optional
        whether to normalize inputs in provided DataProcessor,
        by default False
    encode_output : bool, optional
        whether to normalize outputs in provided DataProcessor,
        by default True
    encoding : str, optional
        parameter for input/output normalization. Whether
        to normalize by channel ("channel-wise") or
        by pixel ("pixel-wise"), default "channel-wise"
    input_subsampling_rate : int or List[int], optional
        rate at which to subsample each input dimension, by default None
    output_subsampling_rate : int or List[int], optional
        rate at which to subsample each output dimension, by default None
    channel_dim : int, optional
        dimension of saved tensors to index data channels, by default 1
    dtype : torch.dtype, optional
        dtype to cast input tensors to after loading. If None, the original
        dtype of the saved data is preserved (e.g. torch.complex64 stays
        complex). Pass torch.float32 to restore the previous default behaviour.
        Default is None.
    channels_squeezed : bool, optional
        If the channels dim is 1, whether that is explicitly kept in the saved tensor.
        If not, we need to unsqueeze it to explicitly have a channel dim.
        Only applies when there is only one data channel, as in our example problems
        Defaults to True

    All datasets are required to expose the following attributes after init:

    train_db: torch.utils.data.Dataset of training examples
    test_db: torch.utils.data.Dataset of test examples
    data_processor: neuralop.data.transforms.DataProcessor to process data examples
        optional, default is None
    """

    def __init__(
        self,
        root_dir: Union[Path, str],
        dataset_name: str,
        n_train: int,
        n_tests: List[int],
        batch_size: int,
        test_batch_sizes: List[int],
        train_resolution: int,
        test_resolutions: List[int],
        encode_input: bool = False,
        encode_output: bool = True,
        encoding="channel-wise",
        input_subsampling_rate=None,
        output_subsampling_rate=None,
        channel_dim=1,
        dtype=None,
        channels_squeezed=True,
    ):
        """Initialize the PTDataset.

        See class docstring for detailed parameter descriptions.
        """

        if isinstance(root_dir, str):
            root_dir = Path(root_dir)

        self.root_dir = root_dir

        # save dataloader properties for later
        self.batch_size = batch_size
        self.test_resolutions = test_resolutions
        self.test_batch_sizes = test_batch_sizes

        # Load train data

        data = torch.load(
        Path(root_dir).joinpath(f"{dataset_name}_train_{train_resolution}.pt").as_posix()
        )

        x_train = data["x"].to(dtype).clone()
        if channels_squeezed:
            x_train = x_train.unsqueeze(channel_dim)

        # optionally subsample along data indices
        ## Input subsampling
        input_data_dims = data["x"].ndim - 2  # batch and channels
        # convert None and 0 to 1
        if not input_subsampling_rate:
            input_subsampling_rate = 1
        if not isinstance(input_subsampling_rate, list):
            # expand subsampling rate along dims if one per dim is not provided
            input_subsampling_rate = [input_subsampling_rate] * input_data_dims
        # make sure there is one subsampling rate per data dim
        assert (
            len(input_subsampling_rate) == input_data_dims
        ), f"Error: length mismatch between input_subsampling_rate and dimensions of data.\
                input_subsampling_rate must be one int shared across all dims, or an iterable of\
                    length {len(input_data_dims)}, got {input_subsampling_rate}"
        # Construct full indices along which to grab X
        train_input_indices = [slice(0, n_train, None)] + [slice(None, None, rate) for rate in input_subsampling_rate]
        train_input_indices.insert(channel_dim, slice(None))
        train_input_indices = tuple(train_input_indices)
        x_train = x_train[train_input_indices]

        y_train = data["y"].clone()
        if channels_squeezed:
            y_train = y_train.unsqueeze(channel_dim)

        ## Output subsampling
        output_data_dims = data["y"].ndim - 2
        # convert None and 0 to 1
        if not input_subsampling_rate:
            output_subsampling_rate = 1
        if not isinstance(output_subsampling_rate, list):
            # expand subsampling rate along dims if one per dim is not provided
            output_subsampling_rate = [output_subsampling_rate] * output_data_dims
        # make sure there is one subsampling rate per data dim
        assert (
            len(output_subsampling_rate) == output_data_dims
        ), f"Error: length mismatch between output_subsampling_rate and dimensions of data.\
                input_subsampling_rate must be one int shared across all dims, or an iterable of\
                    length {len(output_data_dims)}, got {output_subsampling_rate}"

        # Construct full indices along which to grab Y
        train_output_indices = [slice(0, n_train, None)] + [slice(None, None, rate) for rate in output_subsampling_rate]
        train_output_indices.insert(channel_dim, slice(None))
        train_output_indices = tuple(train_output_indices)
        y_train = y_train[train_output_indices]

        del data

        # Fit optional encoders to train data
        # Actual encoding happens within DataProcessor
        if encode_input:
            if encoding == "channel-wise":
                reduce_dims = list(range(x_train.ndim))
                # preserve mean for each channel
                reduce_dims.pop(channel_dim)
            elif encoding == "pixel-wise":
                reduce_dims = [0]

            input_encoder = UnitGaussianNormalizer(dim=reduce_dims)
            input_encoder.fit(x_train)
        else:
            input_encoder = None

        if encode_output:
            if encoding == "channel-wise":
                reduce_dims = list(range(y_train.ndim))
                # preserve mean for each channel
                reduce_dims.pop(channel_dim)
            elif encoding == "pixel-wise":
                reduce_dims = [0]

            output_encoder = UnitGaussianNormalizer(dim=reduce_dims)
            output_encoder.fit(y_train)
        else:
            output_encoder = None

        # Save train dataset
        self._train_db = TensorDataset(
            x_train,
            y_train,
        )

        # create DataProcessor
        self._data_processor = DefaultDataProcessor(
            in_normalizer=input_encoder, out_normalizer=output_encoder
        )

        # load test data
        self._test_dbs = {}
        for res, n_test in zip(test_resolutions, n_tests):
            print(f"Loading test db for resolution {res} with {n_test} samples ")
            data = torch.load(Path(root_dir).joinpath(f"{dataset_name}_test_{res}.pt").as_posix())

            x_test = data["x"].to(dtype).clone()
            if channels_squeezed:
                x_test = x_test.unsqueeze(channel_dim)
            # optionally subsample along data indices
            test_input_indices = [slice(0, n_test, None)] + [slice(None, None, rate) for rate in input_subsampling_rate]
            test_input_indices.insert(channel_dim, slice(None))
            test_input_indices = tuple(test_input_indices)
            x_test = x_test[test_input_indices]

            y_test = data["y"].clone()
            if channels_squeezed:
                y_test = y_test.unsqueeze(channel_dim)
            test_output_indices = [slice(0, n_test, None)] + [slice(None, None, rate) for rate in output_subsampling_rate] 
            test_output_indices.insert(channel_dim, slice(None))
            test_output_indices = tuple(test_output_indices)
            y_test = y_test[test_output_indices]

            del data

            test_db = TensorDataset(
                x_test,
                y_test,
            )
            self._test_dbs[res] = test_db

    @property
    def data_processor(self):
        return self._data_processor

    @property
    def train_db(self):
        return self._train_db

    @property
    def test_dbs(self):
        return self._test_dbs
