import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, SimpleConv, GCNConv,  global_mean_pool
from torch_geometric.nn.dense import Linear
from torch_geometric.nn.aggr import MeanAggregation
from torch_geometric.nn.models import GAT, InnerProductDecoder
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import unbatch
from data.utils import rescale_avg_edge_length
import numpy as np


class NormalizeRotationAndScaleShift(BaseTransform):
    
    def __init__(self, max_points = -1, sort = False):
        self.max_points = max_points
        self.sort = sort
        self.max_scale_value = 10

    def forward(self, batch) :

        positions = list(unbatch(batch.x, batch.batch))
        disp = []
        before = 0
        for i in range(len(positions)):
            pos = positions[i]

            if self.max_points > 0 and pos.size(0) > self.max_points:
                perm = torch.randperm(pos.size(0))
                pos = pos[perm[:self.max_points]]

            pos = pos - pos.mean(dim=0, keepdim=True)
            C = torch.matmul(pos.t(), pos)
            e, v = torch.linalg.eig(C)  # v[:,j] is j-th eigenvector
            e, v = torch.view_as_real(e), v.real

            if self.sort:
                indices = e[:, 0].argsort(descending=True)
                v = v.t()[indices].t()

            pos = torch.matmul(positions[i], v)

            # min, max = pos.aminmax(dim=0, keepdims=True)
            min = pos.amin(dim=0, keepdims=True)
            max = pos.amax(dim=0, keepdims=True)


            size = max - min
            size =  size.squeeze()
            if size[0]>size[1]:
                pos = self.max_scale_value  * (pos-min) / (size[0])
            else:
                pos = self.max_scale_value  * (pos-min) / (size[1])
            

            edge_mask =torch.logical_and(
                batch.batch[batch.edge_index[0]]==i,
                batch.distance_matrix==1
            )
            edges = batch.edge_index[:,edge_mask]
            edges = edges - edges.min()
            avg = torch.linalg.norm(pos[edges[0]]-pos[edges[1]],dim=-1  ).mean()
            pos = pos / avg
            disp.append(pos)

        batch.x = torch.concatenate(disp, dim=0)                
        return batch
    
    