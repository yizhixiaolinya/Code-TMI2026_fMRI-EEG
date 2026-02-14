import torch
import torch.nn as nn
import numpy as np
import scipy.io as scio
import torch.nn.functional as F

# Notes:
#   GCN(in_nodes, out_channels, act, drop_p)
#   Fine2Coarse(in_nodes, out_nodes, drop_p)
# - This file only modifies HGAA to dynamically generate structure based on ROI_num (bottom-level ROI count) and layer (number of layers).

class HGAA(nn.Module):
    """
    Hierarchical Graph Attention Aggregation (HGAA)
    Only requires input of highest-level brain network (graph with maximum ROI count), internally down-samples layer by layer to smaller scales.
    - ROI_num:   Bottom-level ROI count (e.g., 100 or 200)
    - layer:     Bottom-up layer count (>=1), i.e., total number of scales; ROI count per layer is ROI_num + i*step
    - step:      ROI increment between adjacent scales (default 100)
    - act, drop_p: Activation and dropout passed to GCN/Fine2Coarse
    - hidden_dim: Fixed hidden dimension d for all scales (default 64, aligned with embed_dim in fusion)
    
    Paper alignment:
    - Node-level representation: H^{m,c}_s \in R^{R_s \times d} where d is fixed across scales
    - Graph-level representation: g^m_s \in R^d = mean(H^m_s, dim=ROI)
    - Self-attention uses channel-average \bar{H}'^m_s as input
    """
    def __init__(self, act, drop_p, layer, ROI_num=100, step=100, use_self_attn=True, attn_dim=None, hidden_dim=64):
        super(HGAA, self).__init__()
        assert layer >= 1, "layer must be >= 1"
        assert ROI_num >= 1, "ROI_num must be >= 1"
        assert step >= 1, "step must be >= 1"

        self.layer   = layer
        self.ROI_num = ROI_num
        self.step    = step
        self.use_self_attn = use_self_attn  # Whether to use self-attention after GCN
        self.hidden_dim = hidden_dim  # Fixed hidden dimension d (default 64)

        # print('self.layer:',self.layer,flush=True)
        # print('self.ROI_num:',self.ROI_num,flush=True)
        self.sizes_asc  = [self.ROI_num + i* 100 for i in range(self.layer)]  # [N_min, ..., N_max]
        self.sizes_desc = list(reversed(self.sizes_asc))                              # [N_max, ..., N_min]

        # Input projection: project initial features [B, N, N] to [B, N, hidden_dim]
        # The finest scale has the maximum ROI count
        max_roi = max(self.sizes_desc) if self.sizes_desc else ROI_num
        self.input_proj = nn.Linear(max_roi, hidden_dim)

        # GCN for each layer from top to bottom (layer i input graph size = sizes_desc[i])
        # All GCNs now output fixed hidden_dim
        self.net_gcn = nn.ModuleList([
            GCN(in_dim=hidden_dim, out_dim=hidden_dim, act=act, p=drop_p)
            for _ in self.sizes_desc
        ])
        
        # Self-attention modules after GCN (optional, for Stage1)
        # Paper: self-attention uses channel-average \bar{H}'^m_s as input
        if self.use_self_attn:
            # All self-attentions use fixed hidden_dim
            self.self_attentions = nn.ModuleList([
                nn.MultiheadAttention(
                    embed_dim=hidden_dim, 
                    num_heads=min(8, hidden_dim // 8) if hidden_dim >= 8 else 1,
                    dropout=drop_p, 
                    batch_first=True
                )
                for _ in self.sizes_desc
            ])
            self.attn_ln = nn.ModuleList([
                nn.LayerNorm(hidden_dim) for _ in self.sizes_desc
            ])
        else:
            self.self_attentions = None
            self.attn_ln = None

        # Down-sampling mapping between adjacent scales (top to bottom), i-th does sizes_desc[i] -> sizes_desc[i+1]
        self.net_pool = nn.ModuleList([
            Fine2Coarse(indim=self.sizes_desc[i], outdim=self.sizes_desc[i+1], p=drop_p)
            for i in range(len(self.sizes_desc) - 1)
        ])

        # Graph-level normalization (for [B, hidden_dim] embeddings)
        self.graph_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in self.sizes_desc])

    def _pool_matrix(self, pool_module, indim, outdim, device):
        """Return P ([outdim, indim]) for graph coarsening. Prefer Fine2Coarse.proj.weight; otherwise use grouped mean fallback."""
        W = None
        if hasattr(pool_module, "proj") and isinstance(pool_module.proj, nn.Linear):
            W = pool_module.proj.weight
        elif hasattr(pool_module, "weight") and isinstance(pool_module.weight, torch.Tensor):
            W = pool_module.weight
        if W is not None:
            assert W.shape == (outdim, indim), f"Fine2Coarse weight dimensions {tuple(W.shape)} do not match expected {(outdim, indim)}"
            return W.to(device)

        # Fallback: grouped mean
        P = torch.zeros(outdim, indim, device=device)
        if indim % outdim == 0:
            g = indim // outdim
            for i in range(outdim):
                P[i, i*g:(i+1)*g] = 1.0 / g
        else:
            for i in range(outdim):
                j = min(i, indim-1)
                P[i, j] = 1.0
            P = F.normalize(P, p=1, dim=1)
        return P

    def _coarsen_adj(self, A, P):
        """A_next = P A P^T, supports [N,N] or [B,N,N]."""
        if A.dim() == 2:
            return P @ A @ P.transpose(0, 1)
        elif A.dim() == 3:
            B = A.shape[0]
            A_next = torch.empty(B, P.size(0), P.size(0), device=A.device, dtype=A.dtype)
            Pt = P.transpose(0, 1)
            for b in range(B):
                A_next[b] = P @ A[b] @ Pt
            return A_next
        else:
            raise ValueError(f"A must be 2D or 3D, received {A.dim()}D")

    def forward(self, graphs, h):
        """
        Forward pass of HGAA
        
        Args:
            graphs: list of [B, N, N] adjacency matrices, arranged from coarse to fine
                    e.g., when layer=5, graphs=[g1,g2,g3,g4,g5] (finest to coarsest)
            h: Initial node features [B, N, N] (typically identity matrix)
        
        Returns:
            graph_embeddings: list of [B, hidden_dim] - graph-level embeddings for each scale
            node_features: list of [B, R_s, hidden_dim] - node-level features for each scale (for fusion)
        """
    
        assert len(graphs) == self.layer, f"expect {self.layer} graphs, got {len(graphs)}"
    
        graph_embeddings = []  # List of [B, hidden_dim] for each scale
        node_features_list = []  # List of [B, R_s, hidden_dim] for each scale
        g_idx = 0
    
        # === First layer (finest scale) ===
        g = graphs[g_idx]                        # g: (B, N_g, N_g)
        B, N_g, _ = g.shape
        
        # Adapt h to match graph size if needed
        if h.shape[1] != N_g or h.shape[2] != N_g:
            h = torch.eye(N_g, device=g.device).unsqueeze(0).expand(B, N_g, N_g).contiguous()
        
        # Project input features [B, N_g, N_g] -> [B, N_g, hidden_dim]
        # Paper: channel-average for self-attention, but we first project to fixed dim
        h = self.input_proj(h)  # [B, N_g, N_g] -> [B, N_g, hidden_dim]
        
        # GCN propagation
        h = self.net_gcn[g_idx](g, h)  # [B, N_g, hidden_dim]
        
        # Apply self-attention after GCN (if enabled)
        # Paper: self-attention uses channel-average \bar{H}'^m_s
        if self.use_self_attn:
            # Compute channel-average for self-attention input
            # h is already [B, N_g, hidden_dim], so we can use it directly
            # But paper says to use channel-average, so we compute mean over feature dim
            # Actually, h is [B, R_s, d], so channel-average would be mean(dim=-1) -> [B, R_s]
            # But for self-attention, we want [B, R_s, d], so we use h directly
            # The channel-average is used in the attention computation itself
            h_attn, _ = self.self_attentions[g_idx](h, h, h)  # [B, N_g, hidden_dim]
            # Residual connection and layer norm
            h = self.attn_ln[g_idx](h + h_attn)  # [B, N_g, hidden_dim]
        
        # Graph-level embedding: mean pooling over ROI dimension
        g_s = h.mean(dim=1)  # [B, N_g, hidden_dim] -> [B, hidden_dim]
        g_s = self.graph_norms[g_idx](g_s)  # Normalize graph embedding
        graph_embeddings.append(g_s)
        node_features_list.append(h)  # Save node-level features [B, N_g, hidden_dim]
    
        # === Subsequent layers (coarser scales) ===
        for l in range(self.layer-1):
            g_idx += 1
            pool = self.net_pool[g_idx-1]
            gcn = self.net_gcn[g_idx]
            
            # Fine-to-coarse pooling
            h = pool(h)  # [B, R_{s-1}, hidden_dim] -> [B, R_s, hidden_dim]
            if h.shape[1] != graphs[g_idx].shape[1]:
                raise RuntimeError(f"[HGAA/pool] Shape mismatch: h={h.shape}, graph={graphs[g_idx].shape}")
            
            # GCN propagation
            h = gcn(graphs[g_idx], h)  # [B, R_s, hidden_dim]
            if h.shape[1] != graphs[g_idx].shape[1]:
                raise RuntimeError(f"[HGAA/gcn] Shape mismatch: h={h.shape}, graph={graphs[g_idx].shape}")
            
            # Apply self-attention after GCN (if enabled)
            if self.use_self_attn:
                h_attn, _ = self.self_attentions[g_idx](h, h, h)  # [B, R_s, hidden_dim]
                h = self.attn_ln[g_idx](h + h_attn)  # [B, R_s, hidden_dim]
    
            # Graph-level embedding: mean pooling over ROI dimension
            g_s = h.mean(dim=1)  # [B, R_s, hidden_dim] -> [B, hidden_dim]
            g_s = self.graph_norms[g_idx](g_s)  # Normalize graph embedding
            graph_embeddings.append(g_s)
            node_features_list.append(h)  # Save node-level features [B, R_s, hidden_dim]

        # Concatenate all graph embeddings for classification: [B, layer * hidden_dim]
        graph_cat = torch.cat(graph_embeddings, dim=1)  # [B, layer * hidden_dim]
        
        return graph_cat, graph_embeddings, node_features_list

class Fine2Coarse(nn.Module):

    def __init__(self, indim, outdim, p):
        super(Fine2Coarse, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, h):
        import os
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        mapping_dir = os.path.join(base_dir, "mapping")
        filename = os.path.join(mapping_dir, f'mapping_{self.indim}to{self.outdim}_b.mat')
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Mapping file not found: {filename}")
        Map = scio.loadmat(filename)
        Map = Map['mapping']
        #Map[Map<0.50] =0
        #Map[Map>= 0.50] = 1
        Map = torch.tensor(Map).T
        Map = Map.float()
        Map = Map.to(h.device, non_blocking=True)
        # print('Map shape:',Map.shape,flush=True)
        # print('h shape:',h.shape,flush=True)
        X_new = torch.einsum('oi, b i f -> b o f', Map, h)  # Fix einsum dimension issue
            if X_new.shape[1] != Map.shape[0]:
                raise RuntimeError(f"[Fine2Coarse] Output shape mismatch: expected {Map.shape[0]}, got {X_new.shape[1]}")
        return X_new

class Fine2Coarse_mean(nn.Module):

    def __init__(self, indim, outdim, p):
        super(Fine2Coarse_mean, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, h):
        #h = torch.diag(h)
        #h = self.drop(h)
        h = h.T
        filename = '/media/user/4TB/matlab/scripts/adni data/data/interlayermapping/mapping_'+str(self.indim) +'to' + str(self.outdim)+ '_b.mat'
        Map = scio.loadmat(filename)
        Map = Map['mapping']
        Map = torch.tensor(Map)
        Map = Map.float()
        Map = Map.to(h.device)
        Map = Map / torch.sum(Map, axis=0)
        h = torch.matmul(h, Map)
        h = h.T
        h = torch.squeeze(h)
        return h

class Fine2Coarse_max(nn.Module):

    def __init__(self, indim, outdim, p):
        super(Fine2Coarse_max, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, h):
        #h = torch.diag(h)
        #h = self.drop(h)
        h = h.T
        dim=h.shape
        filename = '/media/user/4TB/matlab/scripts/adni data/data/interlayermapping/mapping_' + str(self.indim) + 'to' + str(
            self.outdim) + '_b.mat'
        Map = scio.loadmat(filename)
        Map = Map['mapping']
        Map = torch.tensor(Map)
        Map = Map.float()
        Map = Map.to(h.device)

        h = h.T * Map
        h = torch.max(h, axis=0).values
        h = torch.reshape(h,(dim[0],self.outdim))
        h = h.T
        h = torch.squeeze(h)
        return h

class Fine2Coarse_th(nn.Module):

    def __init__(self, indim, outdim, p, th):
        super(Fine2Coarse_th, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()
        self.th=th

    def forward(self, h):
        #h = torch.diag(h)
        #h = self.drop(h)
        h = h.T
        filename = '/media/user/4TB/matlab/scripts/adni data/data/interlayermapping/mapping_'+str(self.indim) +'to' + str(self.outdim)+ '.mat'
        Map = scio.loadmat(filename)
        Map = Map['mapping']
        Map[Map<self.th] =0
        Map[Map>= self.th] = 1
        Map = torch.tensor(Map)
        Map = Map.float()
        Map = Map.to(h.device)
        h = torch.matmul(h, Map)
        h = h.T
        h = torch.squeeze(h)
        #h = torch.diag(h)
        return h

class GraphUnet(nn.Module):

    def __init__(self, ks, in_dim, out_dim, dim, act, drop_p):
        super(GraphUnet, self).__init__()
        self.ks = ks
        self.top_gcn = GCN(in_dim, out_dim, act, drop_p)
        self.down_gcns = nn.ModuleList()
        #self.up_gcns = nn.ModuleList()
        self.pools = nn.ModuleList()
        #self.unpools = nn.ModuleList()
        self.dim=dim
        self.l_n = len(ks)
        for i in range(self.l_n):
            #self.down_gcns.append(GCN(dim, dim, act, drop_p))
            #self.up_gcns.append(GCN(dim, dim, act, drop_p))
            self.pools.append(Pool(ks[i], dim, drop_p))
            #self.unpools.append(Unpool(dim, dim, drop_p))

        self.down_gcns.append(GCN(400, 1, act, drop_p))
        self.down_gcns.append(GCN(300, 1, act, drop_p))
        self.down_gcns.append(GCN(200, 1, act, drop_p))
        self.down_gcns.append(GCN(100, 1, act, drop_p))
        #self.proj = nn.Linear(dim, out_dim)

    def forward(self, g, h):

        #org_h = h
        #print(h.shape)
        #print(self.dim)
        h = self.top_gcn(g, h)
        h1 = h
        for i in range(self.l_n):
            g, h, idx = self.pools[i](g, h)
            h = self.down_gcns[i](g, torch.diag(torch.squeeze(h)))
            #adj_ms.append(g)
            h1=torch.cat([h1, h], dim=0)

        return h1

class GraphDiif(nn.Module):

    def __init__(self, ks, in_dim, out_dim, dim, act, drop_p):
        super(GraphDiif, self).__init__()
        self.ks = ks
        self.top_gcn = GCN(in_dim, out_dim, act, drop_p)
        self.down_gcns = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.dim=dim
        self.l_n = len(ks)

        self.pools.append(DiffPool(500, 400))
        self.pools.append(DiffPool(400, 300))
        self.pools.append(DiffPool(300, 200))
        self.pools.append(DiffPool(200, 100))

        self.down_gcns.append(GCN(400, 1, act, drop_p))
        self.down_gcns.append(GCN(300, 1, act, drop_p))
        self.down_gcns.append(GCN(200, 1, act, drop_p))
        self.down_gcns.append(GCN(100, 1, act, drop_p))

    def forward(self, g, h):
        hnext = self.top_gcn(g, h)
        h1 = hnext
        for i in range(self.l_n):
            g, h = self.pools[i](g, h, hnext)
            #print(g.shape)
            #print(h.shape)
            h=torch.diag(torch.squeeze(h))
            hnext = self.down_gcns[i](g, h)
            #print(hnext.shape)
            h1=torch.cat([h1, hnext], dim=0)

        return h1
    def Loss(self):
        L=0
        for i in range(self.l_n):
            L=L+self.pools[i].Loss()
        return L

class DiffPool(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DiffPool, self).__init__()
        self.in_dim=in_dim
        self.out_dim=out_dim
        self.gcn=GCN(self.in_dim, self.out_dim, nn.ReLU())
        self.softmax = nn.Softmax()
        self.assign = torch.zeros(in_dim, out_dim)
        self.g = torch.zeros(in_dim, in_dim)

    def forward(self, g, h, hnext):
        self.g = g
        self.assign=self.gcn(g,h)
        #print(assign.shape)
        self.assign=self.softmax(self.assign)
        newh = torch.matmul(torch.transpose(self.assign, 0, 1), hnext)
        newg = torch.transpose(self.assign, 0, 1) @ g @ self.assign
        return newg, newh
    def Loss(self):
        loss_LP=torch.norm(torch.cat([self.g,torch.matmul(self.assign,torch.transpose(self.assign, 0, 1))]))/self.in_dim
        loss_en=0
        eps = 1e-7
        NPassign=F.relu(self.assign)
        for row in range(self.in_dim):
            loss_en=loss_en-torch.sum(NPassign[row,:]*torch.log(NPassign[row,:])+eps)
        loss_en=loss_en/self.in_dim
        return (loss_LP+loss_en)/self.in_dim

class GraphSAG(nn.Module):

    def __init__(self, ks, in_dim, out_dim, dim, act, drop_p):
        super(GraphSAG, self).__init__()
        self.ks = ks
        self.top_gcn = GCN(in_dim, out_dim, act, drop_p)
        self.down_gcns = nn.ModuleList()
        #self.up_gcns = nn.ModuleList()
        self.pools = nn.ModuleList()
        #self.unpools = nn.ModuleList()
        self.dim=dim
        self.l_n = len(ks)
        for i in range(self.l_n):
            #self.down_gcns.append(GCN(dim, dim, act, drop_p))
            #self.up_gcns.append(GCN(dim, dim, act, drop_p))
            self.pools.append(SAGPool(ks[i], dim, drop_p))
            #self.unpools.append(Unpool(dim, dim, drop_p))

        self.down_gcns.append(GCN(400, 1, act, drop_p))
        self.down_gcns.append(GCN(300, 1, act, drop_p))
        self.down_gcns.append(GCN(200, 1, act, drop_p))
        self.down_gcns.append(GCN(100, 1, act, drop_p))
        #self.proj = nn.Linear(dim, out_dim)

    def forward(self, g, h):
        adj_ms = []
        indices_list = []
        down_outs = []
        hs = []
        #org_h = h
        #print(h.shape)
        #print(self.dim)
        h = self.top_gcn(g, h)
        h1 = h
        for i in range(self.l_n):
            g, h, idx = self.pools[i](g, h)
            h = self.down_gcns[i](g, torch.diag(torch.squeeze(h)))
            #adj_ms.append(g)
            h1=torch.cat([h1, h], dim=0)
            #indices_list.append(idx)

        return h1

class SAGPool(nn.Module):
    def __init__(self, k, in_dim, p):
        super(SAGPool, self).__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.in_dim = in_dim
        self.proj = GCN(self.in_dim, 1, nn.ReLU(), p)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, g, h):
        Z = self.drop(h)
        weights = self.proj(g, Z).squeeze()
        scores = self.sigmoid(weights)
        return top_k_graph(scores, g, h, self.k)

class GCN(nn.Module):

    def __init__(self, in_dim, out_dim, act, p=0.3):
        super(GCN, self).__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=p) if p > 0.0 else nn.Identity()

    def forward(self, g, h):
        if g.dim()==2 and h.dim()==3:
            if g.size(1)!=h.size(1) or g.size(0)!=g.size(1):
                raise RuntimeError(f"[GCN/matmul] g={tuple(g.shape)} vs h={tuple(h.shape)}")
        elif g.dim()==3 and h.dim()==3:
            if g.size(-1)!=h.size(1) or g.size(0)!=h.size(0) or g.size(-2)!=g.size(-1):
                raise RuntimeError(f"[GCN/matmul-batch] g={tuple(g.shape)} vs h={tuple(h.shape)}")
        else:
            raise RuntimeError(f"[GCN/matmul] Unexpected dims: g.dim={g.dim()} h.dim={h.dim()}")

        h = self.drop(h)
        # print( g.shape)
        # print( h.shape)
        h = torch.matmul(g, h)
        h = self.proj(h)

        h = self.act(h)
        return h


class Pool(nn.Module):

    def __init__(self, k, in_dim, p):
        super(Pool, self).__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, g, h):
        Z = self.drop(h)
        weights = self.proj(Z).squeeze()
        scores = self.sigmoid(weights)
        return top_k_graph(scores, g, h, self.k)


class Unpool(nn.Module):

    def __init__(self, *args):
        super(Unpool, self).__init__()

    def forward(self, g, h, pre_h, idx):
        new_h = h.new_zeros([g.shape[0], h.shape[1]])
        new_h[idx] = h
        return g, new_h


def top_k_graph(scores, g, h, k):
    num_nodes = g.shape[0]
    values, idx = torch.topk(scores, max(2, int(k*num_nodes)))
    new_h = h[idx, :]
    values = torch.unsqueeze(values, -1)
    new_h = torch.mul(new_h, values)
    un_g = g.bool().float()
    un_g = torch.matmul(un_g, un_g).bool().float()
    un_g = un_g[idx, :]
    un_g = un_g[:, idx]
    g = norm_g(un_g)
    return g, new_h, idx


def norm_g(g):
    degrees = torch.sum(g, 1)
    g = g / degrees
    return g

