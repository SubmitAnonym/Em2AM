import torch_geometric
import torch_geometric.data
from torch_geometric.utils import scatter,remove_self_loops,sort_edge_index,to_dense_adj
import torch
import numpy as np

def get_paths(edge_index,q):
    p = q
    

class KLQ(torch.nn.Module):
    def forward(self,Y, edge_index,distance_matrix):
        diff: torch.Tensor = Y[edge_index[0]]-Y[edge_index[1]]
            # dist1 = torch.clamp((diff.flatten(end_dim=-2)[not_diag_mask.view(-1)]**2).sum(-1),min=1e-16)**0.5
            # dist = torch.linalg.norm(diff.flatten(end_dim=-2)[not_diag_mask.view(-1)],dim=-1)
        dist = torch.linalg.norm(diff,dim=-1)
            #print(sigma.shape,d.shape)
            #p = torch.exp(-d**2/(2*sigma.unsqueeze(-1)**2))
            #p = torch.where(not_diag_mask,p,torch.zeros_like(p)).reshape_as(p)
            #p = p/p.sum(-1,keepdim=True)


        q = ((1+dist**2)**-1)
            # q = (q.view(-1,N-1)/(q.view(-1,N-1).sum(-1,keepdim=True))).view(-1)
        q = (q/q.sum())
        return dist,q


def get_distance_matrix(b):
    if isinstance(b,torch_geometric.data.Batch):
        cur = 0
        DMs = []
        for i in range(len(b.n)):
            n = b.n[i].item()
            DM = torch.zeros((n,n),dtype=torch.int)
            for j in range(n):
                DM[j,0:j] = b.apsp_attr[cur:cur+j]
                DM[j+1:n]=b.apsp_attr[cur+j:cur+n-1]
                cur+=n-1
            DMs.append(DM)    
        return DM
    elif isinstance(b,torch_geometric.data.Data):
        n = b.num_nodes
        DM = torch.zeros((n,n),dtype=torch.int)
        cur=0
        for j in range(n):
            DM[j,0:j] = b.apsp_attr[cur:cur+j]

            DM[j,j+1:n] = b.apsp_attr[cur+j:cur+n-1]
            cur+=n-1
        return DM
def p_ij_conditional_var_dense(X,sigma,test=False):
    tmp1 = -(X**2)
    tmp2 = (2*(sigma**2)).unsqueeze(-1)
    assert_same_shape(tmp1, tmp2)
    sqdist = torch.exp(tmp1/tmp2)
    tmp3 = sqdist
    diag_mask = torch.nn.functional.one_hot(torch.arange(sqdist.shape[-1]),sqdist.shape[-1]).to(device=sqdist.device)
    assert_same_shape(diag_mask,sqdist)
    sqdist = torch.where(diag_mask==1,torch.zeros_like(sqdist),sqdist)

    r =  sqdist/sqdist.sum(-1,keepdim=True)
    if(test and r.isnan().any()):
        raise Exception((sqdist.sum(-1)==0).any(),sigma,(X**2),sqdist[7],tmp3[7],tmp1[7],tmp2[7],torch.argwhere(sqdist.sum(-1)==0))
    return r
def p_ij_conditional_var(data,sigma,test=False):
    tmp1 = -(data.distance_matrix**2)
    tmp2 = (2*(sigma**2)).to(tmp1.device)
    tmp2= tmp2[data.edge_index[0]]
    assert_same_shape(tmp1, tmp2)
    denom = torch.where(tmp1==0,torch.ones_like(tmp2),tmp2)
    sqdist = torch.exp(tmp1/denom)
    sqdist = torch.where(tmp1==0,torch.zeros_like(sqdist),sqdist)
    r = sqdist/(scatter(sqdist,data.edge_index[0],reduce='sum')[data.edge_index[0]])


    return r
def assert_same_shape(tmp1, tmp2):
    assert len(tmp1.shape)==len(tmp2.shape) and [tmp1.shape[i]==tmp2.shape[i] for i in range(len(tmp1.shape))],(tmp1.shape,tmp2.shape)
def find_sigma(data,N,perplexity,sigma_iters,verbose=True):
    perplexity = perplexity.to(data.x.device)
    assert len(perplexity.shape)==0, perplexity.shape
    sigma = torch.ones((N,),device=perplexity.device)
    pij = p_ij_conditional_var(data,sigma)
    sigmin = torch.full_like(sigma,1e-8)
    do_print = False
    while pij.isnan().any():
        do_print = True
        sigmin = torch.where(pij.isnan().any(-1),sigma,sigmin)
        sigma = torch.where(pij.isnan().any(-1),sigma*2,sigma)
    target = torch.log(perplexity).to(data.x.device)
    sigmax = torch.full_like(sigma,np.inf)
    isfinite = torch.ones(sigma.shape,dtype=bool)
    for i in range(sigma_iters) : 
        p_ij = p_ij_conditional_var(data,sigma)
        P = torch.clamp(p_ij,min=1e-20)
        entropy = -scatter(P*torch.log(P),data.edge_index[0])
        # P2 = P.view(N,N)
        # entropy2 = -torch.sum(P2*torch.log(P2),dim=1)
        # assert torch.all((entropy-entropy2)<1e-5)
        if entropy.exp().isnan().any():
            entropy = torch.where(entropy.isnan(),torch.full_like(target,-torch.inf),entropy)
            isfinite = torch.logical_and(isfinite,entropy.isfinite())
            do_print = do_print or True
            #raise Exception(str((X.shape,N,perplexity,P.isnan().any(),torch.log(P).isnan().any(),p_ij.isnan().any(),sigma.isnan().any())))
        sigmin = torch.where(entropy<target,sigma,sigmin)
        sigmax = torch.where(entropy>target,sigma,sigmax)
        newsigma = torch.where(sigmax.isinf(),sigma*2,0.5*(sigmin+sigmax))
        pij = p_ij_conditional_var(data,newsigma)
        do_print = do_print or pij.isnan().any()
        sigmin = torch.where(pij.isnan().any(-1),newsigma,sigmin)
        sigma = torch.where(pij.isnan().any(-1),sigma,newsigma)
    try:
        p_ij_conditional_var(data,sigma,True)
    except Exception as e:
        print(sigma[7],sigmin[7],sigmax[7])
        raise e
    if do_print and verbose:
        print(f"Used debug code for sigma and perplexity={perplexity}")
    return sigma

def find_Y(data:torch_geometric.data.Data,Y:torch.nn.Parameter,sigma,N,n_epochs,initial_lr,final_lr,lr_switch,initial_momentum,
           final_momentum, momentum_switch,
           initial_l_kl, final_l_kl, l_kl_switch,
           initial_l_c, final_l_c, l_c_switch,
           initial_l_r, final_l_r, l_r_switch,
           r_eps=0.05,debug_tf=False,window_size=10,threshold=1e-7,perplexity=None,q_computer=KLQ()):
    pl = torch.nn.ParameterList([Y])
    not_diag_mask = (1-torch.nn.functional.one_hot(torch.arange(Y.shape[0]),Y.shape[0])).bool().to(device=Y.device)
    l_kl = initial_l_kl
    l_c = initial_l_c
    l_r = initial_l_r
    lr = initial_lr
    momentum = initial_momentum
    with torch.no_grad():
        velocity = torch.zeros_like(Y)
    losses = []
    m=0
    p = p_ij_conditional_var(data,sigma)
    # p_dense = p_ij_conditional_var_dense(to_dense_adj(data.edge_index,edge_attr=data.distance_matrix)[0],sigma)
    # assert ((p.reshape(N,N)-p_dense).abs()<1e-5).all(),(p[:40],p_dense[:2],)
    if not p.isfinite().all() or p.isnan().any():
        raise Exception(X,sigma)
    sorted_by_dest = sort_edge_index(data.edge_index,p,sort_by_row=False)
    # assert (sorted_by_dest[0][0]==data.edge_index[1]).all()
    # assert (sorted_by_dest[0][1]==data.edge_index[0]).all()
    p2 = (p+sorted_by_dest[1])/(2*N)
    # p_dense2 = (p_dense+p_dense.T)/(N*2)
    # assert ((p2.reshape(N,N)-p_dense2).abs()<1e-5).all(),(p2[:40],p_dense2[:2],p2[3],p_dense2[0,3],data.edge_index[:,3],sorted_by_dest[0][:,3],p[3],sorted_by_dest[1][3],p_dense[0,3],((p_dense).T)[0,3])
    p=p2
    # p_dense = p_dense2
    edge_index,p = remove_self_loops(data.edge_index,p)

    # p_dense = p_dense.flatten()[not_diag_mask.view(-1)]      
    # assert ((p-p_dense).abs()<1e-5).all(),(p[:40],p_dense[:40],edge_index[:,:40])
    step_sizes = []
    i=1
    while i<=n_epochs:
        if i >= lr_switch:
            lr = final_lr
            step_sizes = []
        if i >= l_r_switch:
            l_r = final_l_r
            step_sizes = []
        if i >= l_c_switch:
            l_c = final_l_c
            step_sizes = []
        if i >= l_kl_switch:
            l_kl = final_l_kl
            step_sizes = []
        if i >= momentum_switch:
            momentum  = final_momentum
            step_sizes = []
        

        dist,q=q_computer(Y,edge_index,data.distance_matrix[data.distance_matrix!=0])
        C_kl = (p*torch.log(torch.clamp(p,min=1e-20)/torch.clamp(q,min=1e-20))).sum()
        l_sum = l_kl +l_c+l_r
        loss_kl = C_kl*(l_kl/l_sum)
        loss_c = l_c/(2*N*l_sum)*torch.linalg.norm(Y,dim=-1).square().sum()
        loss_r = (l_r/l_sum)*(-1/(2*(N**2))) * (torch.clamp(dist,min=1e-8)+r_eps).log().sum()
        l = loss_kl + loss_c+loss_r
        l.backward()
        losses.append(torch.stack((loss_kl.detach(),loss_c.detach(),loss_r.detach())))

        velocity2 = momentum*velocity-lr*Y.grad
        velocity = velocity2
        with torch.no_grad():
          bb = Y.aminmax(dim=0)
          step_sizes.append(torch.linalg.norm(velocity,dim=1).sum()/N*lr*(bb[1]-bb[0]).amax())
        pl.zero_grad()

        with torch.no_grad():
            Y.add_(velocity)
        window_max_step = torch.amax(torch.stack(step_sizes[-window_size:])).item()
        if len(step_sizes)>=window_max_step and window_max_step<threshold:
            switches = np.array(sorted(list(set([lr_switch,l_r_switch,l_c_switch,l_kl_switch,momentum_switch]))))
            if np.any(switches>i):
              newi=switches[switches>i][0]+1    
              step_sizes = []
            else:
              newi=n_epochs+1
            print(f"early convergence {window_max_step} {threshold} {i} {N} {perplexity} {newi}")
            i = newi
        else:
            i+=1
            if i>n_epochs:
              print(f"no convergence {window_max_step} {threshold} {i} {N} {perplexity} ")

    
    return losses
    
def tsnet(data,perplexity,sigma_iters=50,n_epochs=None,initial_lr=None,final_lr=None,lr_switch=None,initial_momentum=None,
           final_momentum=None, momentum_switch=None,
           initial_l_kl=None, final_l_kl=None, l_kl_switch=None,
           initial_l_c=None, final_l_c=None, l_c_switch=None,
           initial_l_r=None, final_l_r=None, l_r_switch=None,Y_init=None,debug_tf=False,return_sigmas=False,q_computer:torch.nn.Module=KLQ()):
    N = data.num_nodes
    if n_epochs is None:
        n_epochs = 1000
    if initial_lr is None:
        initial_lr = 10
    if final_lr is None:
        final_lr = initial_lr
    if lr_switch is None:
        lr_switch = (n_epochs+1)//2
    
    if initial_momentum is None:
        initial_momentum = 0.5
    if final_momentum is None:
        final_momentum = initial_momentum
    if momentum_switch is None:
        momentum_switch = (n_epochs)//2
    
    if initial_l_kl is None:
        initial_l_kl = 1
    if final_l_kl is None:
        final_l_kl = 1
    if l_kl_switch is None:
        l_kl_switch = (n_epochs)//2
    
    if initial_l_c is None:
        initial_l_c = 1.2
    if final_l_c is None:
        final_l_c = 0.01
    if l_c_switch is None:
        l_c_switch = (n_epochs)//2
    
    if initial_l_r is None:
        initial_l_r = 0
    if final_l_r is None:
        final_l_r = 0.6
    if l_r_switch is None:
        l_r_switch = (n_epochs)//2
    if Y_init is None:
        # gen = torch.random.manual_seed(123456789)
        # Y = torch.rand(size=(N,2),dtype=torch.float,device=X.device,generator=gen)
        Y = torch.rand(size=(N,2),dtype=torch.float,device=data.x.device)
    else:
        if isinstance(Y_init,torch.Tensor):
            Y=Y_init.clone()
        elif isinstance(Y_init,np.ndarray):
            Y=torch.as_tensor(Y_init)
        else:
            raise Exception("unexpected type for init")
    #print(Y)
    Y = torch.nn.Parameter(Y)
    sigma = find_sigma(data,N,perplexity,sigma_iters)
    losses = find_Y(data,Y,sigma,N,n_epochs,initial_lr,final_lr,lr_switch,initial_momentum,
           final_momentum, momentum_switch,
           initial_l_kl, final_l_kl, l_kl_switch,
           initial_l_c, final_l_c, l_c_switch,
           initial_l_r, final_l_r, l_r_switch,debug_tf=debug_tf,perplexity=perplexity,q_computer=q_computer)
    if return_sigmas:
        return Y,losses,sigma
    return Y,losses



    
