from matplotlib.pylab import add
import torch
import torch_geometric
import torch_geometric.data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj
import networkx as nx

import numpy as np
import os
import json
from matplotlib import pyplot as plt

import time



def train(model, train_loader, val_loader, loss_fn, supervised=True, device="cuda", epochs=100, train_dir="./train_dir", lr=0.001, additional_loss_fn=None, add_loss_fn_weight=0., add_loss_fn_decay=1., draw_layout=True, draw_matrices=False, draw_loss=True,patience=20):
    
    assert (additional_loss_fn is None and add_loss_fn_weight==0.) or ((add_loss_fn_weight>=0) and add_loss_fn_weight<=1), "If additional loss function is provided, its weight must be >=0 and <=1"
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)    
    # scheduler = None
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.5,patience=patience//2)
    history = {
    "train_loss":[],
    "val_loss":[]
    }
    if scheduler is not None:
        history["lr"]=[]
    model.to(device)
    
    if(os.path.exists(train_dir)):
        os.system(f"rm -rf {train_dir}/*")
    os.makedirs(train_dir, exist_ok=True)
    os.system(f"cp {__file__} {train_dir}/")
    
    def get_last_lr(lastlr):
        if isinstance(lastlr,float):
            return lastlr
        if(isinstance(lastlr,torch.Tensor)):
            return lastlr.mean()
        return np.mean([get_last_lr(l) for l in lastlr])
    for epoch in range(epochs):
        if(scheduler is not None):
            
            history["lr"].append(get_last_lr(scheduler.get_last_lr()))
        main_loss_weight = 1. - add_loss_fn_weight
        model.train()
        train_loss = 0
        main_train_loss = 0
        additional_train_loss = 0
        t = time.perf_counter()
        for i_batch, batch in enumerate(train_loader):
            batch.to(device)
            
            optimizer.zero_grad()
            pred = model(batch)
            # print("==============")
            # print(pred)
            
            loss = loss_fn(pred, batch)
            main_train_loss = main_train_loss + main_loss_weight*loss.item()
            
            if additional_loss_fn is not None:
                addLoss = additional_loss_fn(pred, batch)
                additional_train_loss += add_loss_fn_weight*addLoss.item()
                loss = main_loss_weight*loss + add_loss_fn_weight* addLoss
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            assert (abs(train_loss - (main_train_loss+additional_train_loss))<1E-4), f"{train_loss}, {main_train_loss} {additional_train_loss}"
            # if i_batch%50:
            if time.perf_counter()-t>10:
                t = time.perf_counter()
                print(f"Epoch {epoch} - train_loss {train_loss / (i_batch+1):.5f} ({main_train_loss / (i_batch+1):.5f} / {additional_train_loss / (i_batch+1):.5f}) - val_loss -", end="\r")
        if additional_loss_fn is not None:
            add_loss_fn_weight = add_loss_fn_weight * add_loss_fn_decay
        
        print(f"Epoch {epoch} - train_loss {train_loss / (i_batch+1):.5f} ({main_train_loss / (i_batch+1):.5f} / {additional_train_loss / (i_batch+1):.5f}) - val_loss -", end="\r")
        history["train_loss"].append(train_loss / (i_batch+1))
    



        model.eval()
        val_loss = 0
        main_val_loss = 0
        additional_val_loss = 0
        vi_batch = 0
        with torch.no_grad():
            for vi_batch, batch in enumerate(val_loader):
                batch.to(device)
                pred = model(batch)
                
                loss = loss_fn(pred, batch)
                
                main_val_loss += main_loss_weight*  loss.item()
            
                if additional_loss_fn is not None:
                    addLoss = additional_loss_fn(pred, batch)
                    additional_val_loss += add_loss_fn_weight*addLoss.item()
                    loss = main_loss_weight*loss +  add_loss_fn_weight*addLoss
                val_loss += loss.item()
                assert (abs(val_loss - (main_val_loss+additional_val_loss))<1E-4), f"{val_loss}, {main_val_loss} {additional_val_loss}"
                

        
        print(f"Epoch {epoch} - train_loss {train_loss / (i_batch+1):.5f} ({main_train_loss / (i_batch+1):.5f} / {additional_train_loss / (i_batch+1):.5f}) - val_loss {val_loss / (len(val_loader)):.5f} ({main_val_loss / (len(val_loader)):.5f} / {additional_val_loss / (len(val_loader)):.5f})", end="\r")
        history["val_loss"].append(val_loss / (len(val_loader)))
        print() # space for end of epoch
        if scheduler is not None:
            scheduler.step(history["val_loss"][-1])

        torch.save(model.state_dict(), f"{train_dir}/epoch_{epoch}.pth")
        json.dump(history, open(f"{train_dir}/history.json", "w"))
        
        if draw_loss:
            plt.clf()
            plt.plot([i+1 for i in range(epoch+1)], history["train_loss"], label="train_loss")
            plt.plot([i+1 for i in range(epoch+1)], history["val_loss"], label="val_loss")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{train_dir}/losses.png")
            plt.close()
        
        if draw_layout:
            rand_graph_index = np.random.randint(0, len(val_loader.dataset))
            graphData = val_loader.dataset[rand_graph_index].detach().clone()
            graphData.to(device)
            realAM = to_dense_adj(graphData.edge_index.detach().clone().to(device), edge_attr=graphData.distance_matrix.detach().clone().to(device))[0].to("cpu").detach().numpy()
            realAM[realAM>1] = 0.
            graph = nx.from_numpy_array(realAM)
            loader = DataLoader([graphData], batch_size=1, shuffle=True)
            
            batch = torch_geometric.data.Batch.from_data_list([graphData])
            
            plt.clf()
            pos = model(batch)
            node_size = 10
            title = f"loss: {loss_fn(pos, batch).item():.5f}"
            if additional_loss_fn is not None:
                title += f"\nadditionanl_loss: {additional_loss_fn(pos, batch):.5f}"
            plt.title(title)
            pos = pos.to("cpu").detach().numpy()
            plt.axis("on")
            nx.draw(graph, pos, node_size=node_size, alpha=0.4)
            plt.savefig(f"{train_dir}/drawing.png")
            plt.close()

        if draw_matrices:
            rand_graph_index = np.random.randint(0, len(val_loader.dataset))
            graphData = val_loader.dataset[rand_graph_index].detach().clone()
            graphData.to(device)
            
            DM = to_dense_adj(graphData.edge_index.detach().clone().to(device), edge_attr=graphData.distance_matrix.detach().clone().to(device))[0].to("cpu").detach().numpy()

            batch = torch_geometric.data.Batch.from_data_list([graphData])
            
            sparse_predicted_matrix = model(batch)
            predicted_matrix  = to_dense_adj(batch.edge_index.detach().clone().to(device), edge_attr=sparse_predicted_matrix.detach().clone().to(device))[0].to("cpu").detach().numpy()

            title = f"loss: {loss_fn(sparse_predicted_matrix, batch).item():.5f}"
            plt.clf()
            plt.tight_layout()
            plt.title(title)
            plt.axis("off")
            plt.subplot(1, 2, 1)  # 2 rows, 2 columns, first position
            plt.axis("off")
            plt.imshow(predicted_matrix/predicted_matrix.max(), cmap="viridis")
            plt.subplot(1, 2, 2)  # 2 rows, 2 columns, first position
            plt.axis("off")
            plt.imshow(DM/DM.max(), cmap="viridis")
            plt.axis("off")
            plt.savefig(f"{train_dir}/matrices.png")
            plt.close()
        if np.argmin(np.array(history["val_loss"]))<epoch-patience:
            print(f"early stopping after {patience}")
            break
    return history


