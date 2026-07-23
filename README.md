# Em2AM: A Novel Faithfulness Framework for Graph Drawing Evaluation

This repository was built to share the source code and datasets of the Em2AM framework. It only has research purposes. Please contact the authors if you have any questions about the content.

The code relies on PyTorch and Torch Geometric. Environment versions are available in `requirements.txt` (see below).

# Content

`src` contains the source code to train and evaluate Em2AM. It also includes quality metrics from the literature that were considered in the article.

`grid_approx` and `rome` contain the graphs from the two datasets of the article with the same name. They're stored as Torch Geometric Tensor format. Each file corresponds to a graph. All graph drawing algorithm layouts and all their corresponding metric values are stored as Attributes of the Tensor in each file. Please see the documentation of [torch_geometric.data.Data](https://pytorch-geometric.readthedocs.io/en/2.5.1/generated/torch_geometric.data.Data.html).


# Install and Train

Em2AM was implemented and tested in Python 3.12

`requirements.txt` This environment was used to train the model in the Em2AM framework

`requirements_freeze.txt` This environment contains the result of `pip freeze` during our experiments, giving exact edpendencies and versions used.

`make` to launch the training from its entry point `src/AM_predictor_neighboorhood_growing.py`
