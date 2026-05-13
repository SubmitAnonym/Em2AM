import torch

from torch_geometric.nn import GCNConv
from torch_geometric.data import Batch
from torch_geometric.transforms import Delaunay
from torch_geometric.utils import coalesce,to_undirected
from model.neural_network import NormalizeRotationAndScaleShift



class IncrementallyGrowing(torch.nn.Module):
    def __init__(self, layers = [4],*args, **kwargs):
        super().__init__(*args, **kwargs)
        l = []
        c=2
        for p in layers:
            l.append(torch.nn.Linear(c,p))
            l.append(torch.nn.ReLU())
            c=p
        self.enc_layers = torch.nn.Sequential(*l)
        l.clear()
        self.conv = GCNConv(c,c)
        c=c*2    
        for p in reversed(layers[:-1]):
            l.append(torch.nn.Linear(c,2*p))
            l.append(torch.nn.ReLU())
            c=2*p
        l.append(torch.nn.Linear(c,1))
        l.append(torch.nn.Sigmoid())
        
        self.dec_layers = torch.nn.Sequential(*l)
        self.delaunay = Delaunay()
        self.normalize= NormalizeRotationAndScaleShift()

    def forward(self,batch:Batch):
        batch = self.normalize(batch)
        node_encoding = self.enc_layers(batch.x)
        neighbourhood_index = neighbourhood(batch, self.delaunay)
        neighborhood_encoding = self.conv(node_encoding,neighbourhood_index)

        encoding = torch.concatenate((node_encoding,F.relu(neighborhood_encoding)),-1)
        return self.dec_layers(encoding[batch.edge_index[0]]*encoding[batch.edge_index[1]]).squeeze(-1)
    def load_from(self,smaller:'IncrementallyGrowing'):
        assert len(smaller.enc_layers)<=len(self.enc_layers)
        for i in range(len(smaller.enc_layers)):
            if isinstance(smaller.enc_layers[i],torch.nn.Linear):
                assert isinstance(self.enc_layers[i],torch.nn.Linear)
                layer:torch.nn.Linear = self.enc_layers[i]
                smaller_layer:torch.nn.Linear = smaller.enc_layers[i]
                assert layer.out_features==smaller_layer.out_features or layer.out_features==2*smaller_layer.out_features
                assert layer.in_features==smaller_layer.in_features or layer.in_features==2*smaller_layer.in_features
                mask = torch.zeros_like(layer.weight.data)
                mask[:smaller_layer.weight.shape[0],:smaller_layer.weight.shape[1]] = torch.ones_like(smaller_layer.weight.data)
                w = torch.zeros_like(layer.weight.data)+(mask)*smaller_layer.weight.data.repeat(layer.weight.shape[0]//smaller_layer.weight.shape[0],layer.weight.shape[1]//smaller_layer.weight.shape[1])
                assert len(w.shape)==len(layer.weight.data.shape)
                assert np.all(np.array(w.shape)==np.array(layer.weight.data.shape))
                layer.weight.data= w
                b = torch.concat([smaller_layer.bias,torch.zeros_like(layer.bias.data[smaller_layer.bias.shape[0]:])],dim=0)
                assert len(b.shape)==len(layer.bias.data.shape)
                assert np.all(np.array(b.shape)==np.array(layer.bias.data.shape))
                layer.bias.data = b
        for i in range(len(smaller.enc_layers),len(self.enc_layers)):
            print(self.enc_layers[i])
            if(isinstance(self.enc_layers[i],torch.nn.Linear)):
                print(self.enc_layers[i])
                self.enc_layers[i].bias.data = torch.zeros_like(self.enc_layers[i].bias)
                w = torch.zeros_like(self.enc_layers[i].weight.data)
                w[:smaller_layer.out_features,:smaller_layer.out_features] = torch.eye(smaller_layer.out_features,dtype=w.dtype,device=w.device)

                self.enc_layers[i].weight.data=w
        for i,smaller_layer in enumerate(reversed(smaller.dec_layers)):
            if isinstance(smaller_layer,torch.nn.Linear):
                assert isinstance(self.dec_layers[len(self.dec_layers)-1-i],torch.nn.Linear)
                layer:torch.nn.Linear = self.dec_layers[len(self.dec_layers)-1-i]
                assert layer.out_features==smaller_layer.out_features or layer.out_features==2*smaller_layer.out_features
                assert layer.in_features==smaller_layer.in_features or layer.in_features==2*smaller_layer.in_features
                if(layer.out_features==smaller_layer.out_features):
                    layer.bias.data = smaller_layer.bias.data
                    if(layer.in_features==smaller_layer.in_features):
                        layer.weight.data = smaller_layer.weight.data
                    else:
                        layer.weight.data = torch.concatenate(
                            [
                                smaller_layer.weight.data[:,:smaller_layer.weight.shape[1]//2],
                                torch.zeros_like(smaller_layer.weight.data[:,:smaller_layer.weight.shape[1]//2]),
                                smaller_layer.weight.data[:,smaller_layer.weight.shape[1]//2:],
                                torch.zeros_like(smaller_layer.weight.data[:,smaller_layer.weight.shape[1]//2:]),
                             ],1
                        )
                else:
                    assert layer.in_features==smaller_layer.in_features*2
                    layer.weight.data = torch.concatenate([
                        torch.concatenate([
                            smaller_layer.weight.data[:smaller_layer.weight.shape[0]//2,:smaller_layer.weight.shape[1]//2],
                            torch.zeros_like(smaller_layer.weight.data[:smaller_layer.weight.shape[0]//2,:smaller_layer.weight.shape[1]//2]),
                            smaller_layer.weight.data[:smaller_layer.weight.shape[0]//2,smaller_layer.weight.shape[1]//2:],
                            torch.zeros_like(smaller_layer.weight.data[:smaller_layer.weight.shape[0]//2,smaller_layer.weight.shape[1]//2:]),
                            ],1),
                            torch.zeros_like(smaller_layer.weight.data[:smaller_layer.weight.shape[0]//2]),
                        torch.concatenate([
                            smaller_layer.weight.data[smaller_layer.weight.shape[0]//2:,:smaller_layer.weight.shape[1]//2],
                            torch.zeros_like(smaller_layer.weight.data[smaller_layer.weight.shape[0]//2:,:smaller_layer.weight.shape[1]//2]),
                            smaller_layer.weight.data[smaller_layer.weight.shape[0]//2:,smaller_layer.weight.shape[1]//2:],
                            torch.zeros_like(smaller_layer.weight.data[smaller_layer.weight.shape[0]//2:,smaller_layer.weight.shape[1]//2:]),
                            ],1),
                            torch.zeros_like(smaller_layer.weight.data[smaller_layer.weight.shape[0]//2:]),
                            ],0)
        for i in range(len(self.dec_layers)-len(smaller.dec_layers)):
            layer = self.dec_layers[i]
            if(isinstance(layer,torch.nn.Linear)):
                smaller_layer:torch.nn.Linear = smaller.dec_layers[0]
                assert layer.out_features == smaller_layer.in_features,(layer,smaller_layer)
                layer.bias.data = torch.zeros_like(layer.bias.data)
                w = torch.zeros_like(layer.weight.data)
                w[:smaller_layer.in_features//2,:smaller_layer.in_features//2]=torch.eye(smaller_layer.in_features//2,dtype=w.dtype,device=w.device)
                w[smaller_layer.in_features//2:,smaller_layer.in_features:3*(smaller_layer.in_features//2)]=torch.eye(smaller_layer.in_features//2,dtype=w.dtype,device=w.device)
                layer.weight.data = w
            
        smaller_conv:GCNConv = smaller.conv
        assert self.conv.out_channels>=smaller_conv.out_channels
        assert self.conv.in_channels>=smaller_conv.in_channels
        mask = torch.zeros_like(self.conv.lin.weight.data)
        mask[:smaller_conv.lin.weight.shape[0],:smaller_conv.lin.weight.shape[1]] = torch.ones_like(smaller_conv.lin.weight.data)
        assert self.conv.lin.weight.shape[0]%smaller_conv.lin.weight.shape[0] ==0,(self.conv.lin.weight.shape,smaller_conv.lin.weight.shape)
        assert self.conv.lin.weight.shape[1]%smaller_conv.lin.weight.shape[1] ==0,(self.conv.lin.weight.shape,smaller_conv.lin.weight.shape)
        w = torch.zeros_like(self.conv.lin.weight.data)+(mask)*smaller_conv.lin.weight.data.repeat(self.conv.lin.weight.shape[0]//smaller_conv.lin.weight.shape[0],self.conv.lin.weight.shape[1]//smaller_conv.lin.weight.shape[1])
        assert len(w.shape)==len(self.conv.lin.weight.data.shape)
        assert np.all(np.array(w.shape)==np.array(self.conv.lin.weight.data.shape))

        self.conv.lin.weight.data= w
        b = torch.concat([smaller_conv.bias,torch.zeros_like(self.conv.bias.data[smaller_conv.bias.shape[0]:])],dim=0)
        assert len(b.shape)==len(self.conv.bias.data.shape)
        assert np.all(np.array(b.shape)==np.array(self.conv.bias.data.shape))
        self.conv.bias.data = b
        print(f"load {self} from {smaller}")





def neighbourhood(batch, delaunay:Delaunay):
    if hasattr(batch,"neighbourhood_index") and batch.neighbourhood_index is not None:
        neighbourhood_index = batch.neighbourhood_index
    else:
        s = batch.edge_index.shape[1]
        data = batch.to_data_list()
        for d in data:
            d.pos = d.x.detach()
        data = [delaunay(d) for d in data]
        face = torch.concatenate([d.face+batch.ptr[i] for i,d in enumerate(data)],dim=-1)
        srcs = torch.concatenate((face[0],face[1],face[2]),dim=0)
        tgts = torch.concatenate((face[1],face[2],face[0]),dim=0)
        edge_index = torch.stack((srcs,tgts),dim=0)
        edge_index = coalesce(edge_index)
        edge_index = to_undirected(edge_index)
        batch.neighbourhood_index=edge_index
        neighbourhood_index = batch.neighbourhood_index
        assert s == batch.edge_index.shape[1]
    return neighbourhood_index

        




