
import gc
import json
import resource

import numpy as np
from torch_geometric.loader import DataLoader 
import torch

from data.utils import load_dataset

import os
from model.neighbourhood_context import IncrementallyGrowing
from model.train import train
from metrics.losses import BatchedEdgeList_CrossEntropy2

def select(dataset,layout,shape_graph):
    d = []
    for data in dataset:
        data.x = data.layout_results.view(data.x.shape[0],-1,2)[:,data.layouts.index(layout)]
        shape_edge_index = getattr(data,shape_graph[0])
        shape_edge_index_layout_id = getattr(data,shape_graph[1])
        edge_index = shape_edge_index[:,shape_edge_index_layout_id==data.layouts.index(layout)]
        data["neighbourhood_index"] = edge_index
        d.append(data)
    return d


def loadable(train_dir,n_epochs,patience=20):
    if(os.path.exists(os.path.join(train_dir,"history.json"))):
        with open(os.path.join(train_dir,"history.json"),"rt") as f:
            history = json.load(f)
            epoch = np.argmin(history['val_loss'])
            if(len(history['val_loss'])==n_epochs):
                return epoch
            if epoch+patience<len(history['val_loss']):
                if np.min(history['val_loss'])<90:
                    return epoch
            return None

    else:
        return None

def choose_smaller_narrower(num_layers, width_factor,widths):
    if(width_factor==1):
        if(num_layers>1):
            num_layers_smaller = num_layers-1
            width_factor_smaller = width_factor
        else:
            num_layers_smaller = None
            width_factor_smaller = None
    else:
        if(width_factor//2) in widths:
            num_layers_smaller = num_layers
            width_factor_smaller = [widths]
        else:
            num_layers_smaller = None
            width_factor_smaller = None
    return num_layers_smaller,width_factor_smaller

def choose_smaller_shallower(num_layers, width_factor,widths):
    if(num_layers==1):
        if width_factor//2 in widths:
            num_layers_smaller = 1
            width_factor_smaller = width_factor//2
        else:
            num_layers_smaller = None
            width_factor_smaller = None
    else:
        num_layers_smaller = num_layers-1
        width_factor_smaller = width_factor
    return num_layers_smaller,width_factor_smaller

if __name__ =="__main__":
    max_mem = 25*(1024**3)
    resource.setrlimit(resource.RLIMIT_RSS,(max_mem,max_mem))
    resource.setrlimit(resource.RLIMIT_DATA,(max_mem,max_mem))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    n_features = 2
    batch_size = 32
    n_epochs = 400
    lr = 0.001

    FLOAT = torch.float32


    graph_path= "/data/shared/graphvis-pytorch-geometric_data/grids_with_shape_graphs//"
    assert (os.path.exists(graph_path))
    shape_graph = ("dt_edge_index","dt_layout_id")
    layouts = []
    depths = range(1,7)
    widths = (2**np.arange(5)).tolist()
    widths = [4]
    run = ""
    # run ="_run_2"

    for layout in ['spring','GEM',"sgd2","LinLog","init", 'PivotMDS', 'FM**3', 'KK', 'Random','tsNET','DNN2']:
        if any([not loadable( f"./train_dir/AM_predictor_neighbourhood_{layout}_growing_bis{run}_{num_layers}_{width_factor}_{shape_graph[0]}",n_epochs) for num_layers in depths for width_factor in widths]):
            layouts.append(layout)
    print(layouts)
    for layout in layouts:

        print("Loading dataset...", flush=True)
        train_dataset = load_dataset(path=f"{graph_path}/train", device=device)
        print(f"loaded train: {len(train_dataset)} samples", flush=True)
        val_dataset = load_dataset(path=f"{graph_path}/val", device=device)
        print(f"loaded val: {len(val_dataset)} samples")
        train_dataset  = select(train_dataset,layout,shape_graph)
        val_dataset  = select(val_dataset,layout,shape_graph)



        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


        loss = BatchedEdgeList_CrossEntropy2()
        loss.to(device)

        models = {i:{} for i in depths}
        base = IncrementallyGrowing()

        for num_layers in depths:
            for width_factor in widths:
                model = IncrementallyGrowing(layers=[base.enc_layers[0].out_features*(2**j)*width_factor for j in range(num_layers)])
                train_dir = f"./train_dir/AM_predictor_neighbourhood_{layout}_growing_bis{run}_{num_layers}_{width_factor}_{shape_graph[0]}"
                epoch = loadable(train_dir,n_epochs)
                if(epoch is not None):
                    model.load_state_dict(torch.load(os.path.join(train_dir,f"epoch_{epoch}.pth")))
                    model.cpu()
                    models[num_layers][width_factor]=model
                else:
                    num_layers_smaller, width_factor_smaller = choose_smaller_shallower(num_layers, width_factor,widths)
                    if(num_layers_smaller is not None and width_factor_smaller is not None):
                        if(num_layers_smaller in models and width_factor_smaller in models[num_layers_smaller]):
                            smaller = models[num_layers_smaller][width_factor_smaller]
                            model.load_from(smaller)
                        else:
                            print(f"smaller model {num_layers_smaller} {width_factor_smaller} is not available to initialize {num_layers} {width_factor}")
                            continue
                    model.compile()
                    model.to(device)
                    print(model)
                    try:
                        while not loadable(train_dir,400):
                            history = train(model, train_loader, val_loader, loss, train_dir=train_dir, supervised=False, device=device, lr=lr, epochs=n_epochs, draw_layout=False)

                        model.cpu()
                        models[num_layers][width_factor]=model
                        gc.collect()
                        torch.cuda.memory.empty_cache()
                    except torch.OutOfMemoryError as e:
                        print(e)
                        print(f"failed to train {num_layers} {width_factor}")
                        continue
        del models
        del train_dataset
        del val_dataset
        del base
        del model
        del loss
        del train_loader
        del val_loader
        gc.collect()
        torch.cuda.memory.empty_cache()

        
