# Em2AM: A Novel Faithfulness Framework for Graph Drawing Evaluation

This repository was built to share the source code and datasets of the Em2AM framework. It only has research purposes. Please contact the authors if you have any questions about the content.

If the repository is too large to be cloned, please clone first commit (without datasets).

The code relies on PyTorch and Torch Geometric. Environment versions are available in `requirements.txt`.

# Content

`src` contains the source code to train and evaluate Em2AM. It also includes quality metrics from the literature that were considered in the article.

`grid_approx` and `rome` contain the graphs from the two datasets of the article with the same name. They're stored as Torch Geometric Tensor format. Each file corresponds to a graph. All graph drawing algorithm layouts and all their corresponding metric values are stored as Attributes of the Tensor in each file. Please see the documentation of [torch_geometric.data.Data](https://pytorch-geometric.readthedocs.io/en/2.5.1/generated/torch_geometric.data.Data.html).


# Environments
`requirements.txt` This environment was used to train the model in the Em2AM framework

`requirements_sgd2.txt` This environment, different from the first one was used to generate s_gd2 layout due to an incompatiblity between s_dg2 and numpy>=2.