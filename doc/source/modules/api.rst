.. _api_ref:

#############
API reference
#############

:mod:`neuralop`: Neural Operators in Python

.. automodule:: neuralop.models
    :no-members:
    :no-inherited-members:

.. _neuralop_models_ref:

======
Models
======

In :mod:`neuralop.models`, we provide neural operator models you can directly use on your applications.

.. _fno_api:

FNO
---

We provide a general Fourier Neural Operator (FNO) that supports most usecases.

It works for any dimension, which is inferred based on `n_modes`
(a tuple with the number of modes to keep in the Fourier domain for each dimension.)

.. autosummary::
    :toctree: generated
    :template: class.rst

    FNO

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _tfno_api:

Tensorized FNO (TFNO)
---------------------

.. autosummary::
    :toctree: generated
    :template: class.rst

    TFNO

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _sfno_api:

Spherical Fourier Neural Operators (SFNO)
-----------------------------------------

.. autosummary::
    :toctree: generated
    :template: class.rst

    SFNO

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _gino_api:

Geometry-Informed Neural Operators (GINO)
-----------------------------------------

.. autosummary::
    :toctree: generated
    :template: class.rst

    GINO

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _local_no_api:

Local Neural Operators (LocalNO)
--------------------------------

.. autosummary::
    :toctree: generated
    :template: class.rst

    LocalNO

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _uno_api:

U-shaped Neural Operators (U-NO)
--------------------------------

.. autosummary::
    :toctree: generated
    :template: class.rst

    UNO

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _rno_api:

Recurrent Neural Operators (RNO)
--------------------------------

.. autosummary::
    :toctree: generated
    :template: class.rst

    RNO

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _otno_api:

Optimal Transport Neural Operators (OTNO)
-----------------------------------------

.. autosummary::
    :toctree: generated
    :template: class.rst

    OTNO

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _uqno_api:

Uncertainty Quantification Neural Operators (UQNO)
--------------------------------------------------

.. autosummary::
    :toctree: generated
    :template: class.rst
    
    UQNO

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _fnogno_api:

Fourier/Geometry Neural Operators (FNOGNO)
------------------------------------------

.. autosummary::
    :toctree: generated
    :template: class.rst
    
    FNOGNO

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _codano_api:

Codomain Attention Neural Operators (CODANO)
--------------------------------------------

.. autosummary::
    :toctree: generated
    :template: class.rst
    
    CODANO

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <hr style="margin: 4em 0; border: none; border-top: 2px solid #e0e0e0;">

.. _neuralop_layers_ref:

======
Layers
======

.. automodule:: neuralop.layers
    :no-members:
    :no-inherited-members:



In addition to the full architectures, we also provide 
in :mod:`neuralop.layers` building blocks,
in the form of PyTorch layers, that you can use to build your own models:

.. _fno_blocks_api:

FNO Blocks
----------

.. automodule:: neuralop.layers.fno_block
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: generated
    :template: class.rst

    FNOBlocks

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _rno_block_api:

RNO Blocks
----------

.. automodule:: neuralop.layers.rno_block
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: generated
    :template: class.rst

    RNOBlock

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _spectral_conv_api:

Fourier Convolutions
--------------------
.. automodule:: neuralop.layers.spectral_convolution
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: generated
    :template: class.rst

    SpectralConv

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _spherical_conv_api:

Spherical Convolutions
----------------------

.. automodule:: neuralop.layers.spherical_convolution
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: generated
    :template: class.rst

    SphericalConv

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. _gno_api:

Graph convolutions and kernel integration
-----------------------------------------

.. automodule:: neuralop.layers.gno_block
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: generated
    :template: class.rst

    GNOBlock

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: neuralop.layers.integral_transform
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: generated
    :template: class.rst

    IntegralTransform

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _local_no_blocks_api:

Local NO Blocks
---------------
.. automodule:: neuralop.layers.local_no_block
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: generated
    :template: class.rst

    LocalNOBlocks

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _diff_conv_api:

Local Integral/Differential Convolutions
----------------------------------------

.. automodule:: neuralop.layers.differential_conv
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: generated
    :template: class.rst

    FiniteDifferenceConvolution

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _disco_conv_api:

Discrete-Continuous (DISCO) Convolutions
----------------------------------------

.. automodule:: neuralop.layers.discrete_continuous_convolution
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: generated
    :template: class.rst

    DiscreteContinuousConv2d
    DiscreteContinuousConvTranspose2d
    EquidistantDiscreteContinuousConv2d
    EquidistantDiscreteContinuousConvTranspose2d

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Codomain Attention (Transformer) Blocks
---------------------------------------
.. automodule:: neuralop.layers.coda_layer
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: generated
    :template: class.rst

    CODALayer

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Channel MLP
-----------

.. automodule:: neuralop.layers.channel_mlp
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: generated
    :template: class.rst

    ChannelMLP

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Embeddings
----------

Apply positional embeddings as additional channels on a function:

.. automodule:: neuralop.layers.embeddings
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: generated
    :template: class.rst

    GridEmbeddingND
    GridEmbedding2D
    SinusoidalEmbedding

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Neighbor search
---------------

Find neighborhoods on arbitrary coordinate meshes:

.. automodule:: neuralop.layers.neighbor_search
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: generated
    :template: class.rst

    NeighborSearch

.. autosummary::
    :toctree: generated
    :template: function.rst

    native_neighbor_search

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Domain Padding
--------------

.. automodule:: neuralop.layers.padding
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: generated
    :template: class.rst

    DomainPadding

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Skip Connections
----------------

.. automodule:: neuralop.layers.skip_connections
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: generated
    :template: function.rst

    skip_connection

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Normalization Layers
--------------------

.. automodule:: neuralop.layers.normalization_layers
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: generated
    :template: class.rst

    AdaIN
    InstanceNorm
    BatchNorm

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Complex-value Support
---------------------

.. automodule:: neuralop.layers.complex
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: generated
    :template: class.rst

    ComplexValued

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <hr style="margin: 4em 0; border: none; border-top: 2px solid #e0e0e0;">


=================
Model Dispatching
=================
We provide a utility function to create model instances from a configuration.
It has the advantage of doing some checks on the parameters it receives.

.. automodule:: neuralop.models.base_model
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: generated
    :template: function.rst

    get_model
    available_models

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <hr style="margin: 4em 0; border: none; border-top: 2px solid #e0e0e0;">

========
Training
========
We provide functionality that automates the boilerplate code associated with 
training a machine learning model to minimize a loss function on a dataset:

.. automodule:: neuralop.training
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: generated
    :template: class.rst

    Trainer
    IncrementalFNOTrainer

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Optimizers
----------

.. automodule:: neuralop.training.tensorgrad
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: generated
    :template: class.rst

    TensorGRaD
    TensorGRaDProjector
    UnstructuredSparseProjector

.. autosummary::
    :toctree: generated
    :template: function.rst

    fno_tensorgrad_param_groups

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Training Utilities
------------------

.. automodule:: neuralop.training.torch_setup
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: generated
    :template: function.rst

    setup

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Multi-Grid Patching
-------------------

.. automodule:: neuralop.training.patching
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: generated
    :template: class.rst

    MultigridPatching2D

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <hr style="margin: 4em 0; border: none; border-top: 2px solid #e0e0e0;">

==============
Loss Functions
==============

Data Losses
-----------

.. automodule:: neuralop.losses.data_losses
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: generated
    :template: class.rst

    LpLoss
    H1Loss
    HdivLoss
    PointwiseQuantileLoss
    MSELoss

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Equation Losses
---------------

Physics-informed loss functions:

.. automodule:: neuralop.losses.equation_losses
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: generated
    :template: class.rst

    BurgersEqnLoss
    ICLoss
    PoissonInteriorLoss
    PoissonBoundaryLoss
    PoissonEqnLoss

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Meta Losses
-----------

Meta-losses for weighting composite loss functions.

.. automodule:: neuralop.losses.meta_losses
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: generated
    :template: class.rst

    WeightedSumLoss
    Relobralo
    SoftAdapt

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Differentiation
---------------

Numerical differentiation utilities:

.. automodule:: neuralop.losses.differentiation
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: generated
    :template: class.rst

    FourierDiff
    FiniteDiff

.. autosummary::
    :toctree: generated
    :template: function.rst

    non_uniform_fd

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Spectral Projection
-------------------

Spectral projection utilities for enforcing physical constraints:

.. automodule:: neuralop.layers.spectral_projection
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: generated
    :template: function.rst

    spectral_projection_divergence_free

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <hr style="margin: 4em 0; border: none; border-top: 2px solid #e0e0e0;">

====
Data
====

In `neuralop.data`, we provide APIs for standardizing PDE datasets (`.datasets`) and transforming raw data into model inputs (`.transforms`).

Datasets
--------

We ship a small dataset for testing:

.. automodule:: neuralop.data.datasets
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: generated
    :template: function.rst

    load_darcy_flow_small

We provide downloadable datasets for Darcy-Flow, Navier-Stokes, and Car-CFD. 

.. automodule:: neuralop.data.datasets.darcy
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: generated
    :template: class.rst

    DarcyDataset


.. automodule:: neuralop.data.datasets.navier_stokes
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: generated
    :template: class.rst

    NavierStokesDataset


.. automodule:: neuralop.data.datasets.car_cfd_dataset
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: generated
    :template: class.rst

    CarCFDDataset

.. _burgers_dataset_api:

.. automodule:: neuralop.data.datasets.burgers
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: generated
    :template: class.rst

    Burgers1dTimeDataset

.. autosummary::
    :toctree: generated
    :template: function.rst

    load_mini_burgers_1dtime


.. note::
    Additional datasets are available with optional dependencies:
    
    * **The Well Datasets**: Large-scale collection of diverse physics simulations
      (requires `the_well` package)
    * **Spherical Shallow Water Equations**: For spherical coordinate systems
      (requires `torch_harmonics` package)
    
    These datasets are conditionally imported and may not be available
    depending on your installation.


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DataProcessors
--------------

Much like PyTorch's `Torchvision.Datasets` module, our `data` module also includes
utilities to transform data from its raw form into the form expected by models and 
loss functions:

.. automodule:: neuralop.data.transforms.data_processors
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: generated
    :template: class.rst

    DefaultDataProcessor
    MGPatchingDataProcessor
    IncrementalDataProcessor

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Normalizers
-----------

Data normalization utilities:

.. automodule:: neuralop.data.transforms.normalizers
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: generated
    :template: class.rst

    UnitGaussianNormalizer
    DictUnitGaussianNormalizer

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <hr style="margin: 4em 0; border: none; border-top: 2px solid #e0e0e0;">

=================
Utility Functions
=================

.. automodule:: neuralop.utils
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: generated
    :template: function.rst

    count_model_params
    count_tensor_params
    spectrum_2d
    compute_rank
    compute_stable_rank
    compute_explained_variance
