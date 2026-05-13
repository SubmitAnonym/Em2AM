from math import tan
import torch
import torch.nn as nn
from torch_geometric.utils import scatter, remove_self_loops







class BatchedEdgeList_CrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred_edge_attr, batch):
        
        assert torch.isfinite(pred_edge_attr).all()
        assert torch.all(pred_edge_attr >= 0) and torch.all(pred_edge_attr<=1)
        
        inputs = pred_edge_attr # predicted values for each edge in edge_index
        target = torch.where(batch.distance_matrix==1,torch.ones_like(batch.distance_matrix),torch.zeros_like(batch.distance_matrix)) # 1 if actual edge else 0 
        num_edges = scatter(target, batch.batch[batch.edge_index[0]], reduce='sum') # num actual edge in each graph in the batch
        complete_no_loop,_ = remove_self_loops(batch.edge_index) # remove the self loops 
        num_edges_complete_no_loop = scatter(torch.ones([complete_no_loop.shape[1]],device=inputs.device,dtype=inputs.dtype),batch.batch[complete_no_loop[0]],reduce='sum') # number of edges in each complete graph minus the self loops
        graph_alpha =  num_edges/num_edges_complete_no_loop # a value of alpha for each graph
        graph_alpha = torch.sqrt(graph_alpha)
        edge_alpha = graph_alpha[batch.batch[batch.edge_index[0]]] # each edge get the alpha value of its graph
        weights = torch.where(target==1,1-edge_alpha,edge_alpha)
        
        assert torch.all(target >= 0) and torch.all(target<=1)
        assert torch.all(weights >= 0) and torch.all(weights<=1)

        # return torch.nn.functional.binary_cross_entropy(inputs, target.float(), weights,reduction='mean')

        bce = torch.nn.functional.binary_cross_entropy(inputs, target.float(), weights, reduction='none')
        bce_graph = scatter(bce,batch.batch[batch.edge_index[0]],reduce='mean')#mean per graph
        return torch.mean(bce_graph)

class BatchedEdgeList_CrossEntropy2(nn.Module):
    def __init__(self,reduction='mean'):
        super().__init__()
        match reduction:
            case 'mean':
                self.reduction = torch.mean
            case 'sum':
                self.reduction = torch.sum
            case 'none':
                self.reduction = lambda a:a
            case _:
                raise Exception(f"unexpected reduction {reduction}. Must be one of 'none', 'sum', 'mean'")

    def forward(self, pred_edge_attr, batch):
        
        assert torch.isfinite(pred_edge_attr).all()
        assert torch.all(pred_edge_attr >= 0) and torch.all(pred_edge_attr<=1)
        
        inputs = pred_edge_attr # predicted values for each edge in edge_index
        target = torch.where(batch.distance_matrix==1,torch.ones_like(batch.distance_matrix),torch.zeros_like(batch.distance_matrix)) # 1 if actual edge else 0 
        # num_edges = scatter(target, batch.batch[batch.edge_index[0]], reduce='sum') # num actual edge in each graph in the batch
        # complete_no_loop,_ = remove_self_loops(batch.edge_index) # remove the self loops 
        # num_edges_complete_no_loop = scatter(torch.ones([complete_no_loop.shape[1]],device=inputs.device,dtype=inputs.dtype),batch.batch[complete_no_loop[0]],reduce='sum') # number of edges in each complete graph minus the self loops
        # graph_alpha =  num_edges/num_edges_complete_no_loop # a value of alpha for each graph
        # graph_alpha = torch.sqrt(graph_alpha)
        # edge_alpha = graph_alpha[batch.batch[batch.edge_index[0]]] # each edge get the alpha value of its graph        
        # weights = torch.where(target==1,1-edge_alpha,edge_alpha)
        edges,inputs = remove_self_loops(batch.edge_index,inputs)
        edges,target = remove_self_loops(batch.edge_index,target)
        positives = scatter(target,batch.batch[edges[0]])
        totals = scatter(torch.ones_like(target),batch.batch[edges[0]])
        weight_positive = 1/positives
        weight_negative = 1/(totals-positives)
        weights = torch.where(target==1,weight_positive[batch.batch[edges[0]]],weight_negative[batch.batch[edges[0]]],)
        
        
        assert torch.all(target >= 0) and torch.all(target<=1)
        assert torch.all(weights >= 0) and torch.all(weights<=1)

        # return torch.nn.functional.binary_cross_entropy(inputs, target.float(), weights,reduction='mean')

        bce = torch.nn.functional.binary_cross_entropy(inputs, target.float(), weights, reduction='none')
        bce_graph = scatter(bce,batch.batch[edges[0]],reduce='sum')#sum per graph
        return self.reduction(bce_graph)

