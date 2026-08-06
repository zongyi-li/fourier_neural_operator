.. image:: https://img.shields.io/pypi/v/neuraloperator
   :target: https://pypi.org/project/neuraloperator/
   :alt: PyPI

.. image:: https://github.com/NeuralOperator/neuraloperator/actions/workflows/test.yml/badge.svg
   :target: https://github.com/NeuralOperator/neuraloperator/actions/workflows/test.yml


###############################################
NeuralOperator: Learning in Infinite Dimensions
###############################################

NeuralOperator is a comprehensive PyTorch library for learning neural operators,
containing the official implementation of Fourier Neural Operators and other neural operator architectures.

NeuralOperator is part of the PyTorch Ecosystem, check the PyTorch `announcement <https://pytorch.org/blog/neuraloperatorjoins-the-pytorch-ecosystem>`_! 


Unlike regular neural networks, neural operators enable learning mapping between function spaces, 
and this library provides all of the tools to do so on your own data. Neural operators are 
resolution invariant, so your trained operator can be applied on data of any resolution.

Checkout the `documentation <https://neuraloperator.github.io/dev/index.html>`_ and our `practical guide <https://arxiv.org/abs/2512.01421>`_ for more information!

============
Installation
============

Just clone the repository and install locally (in editable mode so changes in the code are 
immediately reflected without having to reinstall):

.. code::

  git clone https://github.com/NeuralOperator/neuraloperator
  cd neuraloperator
  pip install -e .
  pip install -r requirements.txt

You can also pip install the most recent stable release of the library 
on `PyPI <https://pypi.org/project/neuraloperator/>`_:


.. code::

  pip install neuraloperator


==========
Quickstart
==========

After you have installed the library, you can start training operators seamlessly:

.. code-block:: python

   from neuralop.models import FNO

   operator = FNO(n_modes=(64, 64), 
                  hidden_channels=64,
                  in_channels=2, 
                  out_channels=1)

Tensorization is also available: you can improve the previous models
by simply using a Tucker Tensor FNO with fewer parameters:

.. code-block:: python

   from neuralop.models import TFNO

   operator = TFNO(n_modes=(64, 64), 
                   hidden_channels=64,
                   in_channels=2, 
                   out_channels=1,
                   factorization='tucker',
                   implementation='factorized',
                   rank=0.1)

This will use a Tucker factorization of the weights. The forward pass
will be efficient by contracting directly the inputs with the factors
of the decomposition. The Fourier layers will have 10% of the parameters
of an equivalent, dense Fourier Neural Operator!

To use W&B logging features, simply create a file in ``neuraloperator/config`` 
called ``wandb_api_key.txt`` and paste your W&B API key there.

Hyperparameter optimization (HPO) scripts using `HyperNOs` and `Ray Tune` are also available in the ``scripts/`` directory (e.g. ``scripts/ray_tune_tfno_darcy.py``).


============
Contributing
============

NeuralOperator is 100% open-source, and we welcome contributions from the community! 

Our mission for NeuralOperator is to provide access to well-documented, robust implementations of 
neural operator methods from foundations to the cutting edge, including new architectures, meta-algorithms, training methods and benchmark datasets. 
We are also interested in integrating interactive examples that showcase operator 
learning in action on small sample problems.

If your work provides one of the above, we would be thrilled to integrate it into the library. 
Otherwise, if your work simply relies on a version of the NeuralOperator codebase, we recommend 
publishing your code in a separate repository. 

If you spot a bug or would like to see a new feature,
please report it on our `issue tracker <https://github.com/neuraloperator/neuraloperator/issues>`_
or open a `Pull Request <https://github.com/neuraloperator/neuraloperator/pulls>`_. 

For detailed development setup, testing, and contribution guidelines, please refer to our `Contributing Guide <CONTRIBUTING.md>`_.


===============
Code of Conduct
===============

All participants are expected to uphold the `Code of Conduct <https://github.com/neuraloperator/neuraloperator/blob/main/CODE_OF_CONDUCT.md>`_ to ensure a friendly and welcoming environment for everyone.


=====================
Citing NeuralOperator
=====================

If you use NeuralOperator in an academic paper, please cite [1]_::

   @article{kossaifi2025librarylearningneuraloperators,
      author    = {Jean Kossaifi and
                     Nikola Kovachki and
                     Zongyi Li and
                     David Pitt and
                     Miguel Liu-Schiaffini and
                     Robert Joseph George and
                     Boris Bonev and
                     Kamyar Azizzadenesheli and
                     Julius Berner and
                     Valentin Duruisseaux and
                     Anima Anandkumar},
      title     = {A Library for Learning Neural Operators},
      journal   = {arXiv preprint arXiv:2412.10354},
      year      = {2025},
   }

and consider citing [2]_ and [3]_::

   @article{duruisseaux2025guide,
      author    = {Valentin Duruisseaux and
                     Jean Kossaifi and
                     Anima Anandkumar},
      title     = {Fourier Neural Operators Explained: A Practical Perspective},
      journal   = {arXiv preprint arXiv:2512.01421},
      year      = {2025},
   }

   @article{kovachki2023neuraloperator,
      author    = {Nikola Kovachki and
                     Zongyi Li and
                     Burigede Liu and
                     Kamyar Azizzadenesheli and
                     Kaushik Bhattacharya and
                     Andrew Stuart and
                     Anima Anandkumar},
      title     = {Neural Operator: Learning Maps Between Function Spaces with Applications to PDEs},
      journal   = {JMLR},
      volume    = {24},
      number    = {1},
      articleno = {89},
      numpages  = {97},
      year      = {2023},
   }


.. [1] Kossaifi, J., Kovachki, N., Li, Z., Pitt, D., Liu-Schiaffini, M., Duruisseaux, V., George, R., Bonev, B., Azizzadenesheli, K., Berner, J., and Anandkumar, A., "A Library for Learning Neural Operators", 2025. https://arxiv.org/abs/2412.10354.

.. [2] Duruisseaux, V., Kossaifi, J., and Anandkumar, A., "Fourier Neural Operators Explained: A Practical Perspective", 2025. https://arxiv.org/abs/2512.01421.

.. [3] Kovachki, N., Li, Z., Liu, B., Azizzadenesheli, K., Bhattacharya, K., Stuart, A., and Anandkumar, A., “Neural Operator: Learning Maps Between Function Spaces with Applications to PDEs”, JMLR, 24(1):89, 2023. https://arxiv.org/abs/2108.08481.
