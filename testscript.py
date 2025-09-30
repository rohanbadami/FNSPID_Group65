# src/data/finbert_embed.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch, pandas as pd, numpy as np
from pathlib import Path

class FinBertEncoder:
    def __init__(self, device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device).eval()
        self.device = device

    @torch.inference_mode()
    def encode_batch(self, texts: list[str]) -> np.ndarray:
        toks = self.tokenizer(texts, truncation=True, padding=True, max_length=256, return_tensors="pt").to(self.device)
        logits = self.model(**toks).logits          # [B, 3] (neg/neu/pos)
        probs  = logits.softmax(-1)                 # use probs as low-dim sentiment feature
        return probs.detach().cpu().numpy()         # shape [B,3]

def build_daily_embeddings(news_df, out_dir):
    """
    news_df columns: [date, ticker, text] (aligned to calendar in calendar_align.py)
    Writes one Parquet per ticker with columns: date, p_neg, p_neu, p_pos and optional pooled CLS.
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    enc = FinBertEncoder()
    for (ticker, grp) in news_df.groupby("ticker"):
        # batch encode, groupby date, mean-pool
        daily = []
        for d, dg in grp.groupby("date"):
            arr = []
            for i in range(0, len(dg), 64):
                arr.append(enc.encode_batch(dg["text"].iloc[i:i+64].tolist()))
            probs = np.concatenate(arr, 0).mean(0)
            daily.append((d, *probs))
        out = pd.DataFrame(daily, columns=["date", "p_neg", "p_neu", "p_pos"]).sort_values("date")
        out.to_parquet(out_dir / f"{ticker}.parquet", index=False)


# src/data/graph_builder.py
import pandas as pd, numpy as np
from pathlib import Path

def rolling_corr_edges(returns_df: pd.DataFrame, tickers: list[str], window=60, thr=0.6):
    """
    returns_df: index=date, columns=tickers, values=log returns
    For the last date in the month, compute corr over trailing `window`
    """
    sub = returns_df.iloc[-window:]
    C = sub.corr().values
    edges = []
    for i, a in enumerate(tickers):
        for j in range(i+1, len(tickers)):
            if abs(C[i,j]) >= thr:
                edges.append((a, j, "corr", float(C[i,j])))
    return edges

def co_mention_edges(co_counts: pd.DataFrame, thr=3):
    """
    co_counts: columns=['src','dst','count'] for current month
    """
    return [(r.src, r.dst, "com", int(r["count"])) for _, r in co_counts[co_counts["count"]>=thr].iterrows()]

def sector_edges(meta_df: pd.DataFrame):
    """
    meta_df: ['ticker','sector'] for tickers active this month
    """
    d = {}
    for sec, g in meta_df.groupby("sector"):
        arr = g["ticker"].tolist()
        for i in range(len(arr)):
            for j in range(i+1, len(arr)):
                d[(arr[i], arr[j])] = ("sec", 1.0)
    return [(a,b,"sec",w) for (a,b),(lab,w) in d.items()]


# src/data/graph_dataset.py
import torch
from torch_geometric.data import InMemoryDataset, Data
from pathlib import Path
import pandas as pd, numpy as np

class MonthlyGraphDataset(InMemoryDataset):
    def __init__(self, graphs_dir: str, seq_window=50, pred_horizon=3, transform=None, pre_transform=None):
        self.graphs_dir = Path(graphs_dir)
        self.seq_window = seq_window
        self.pred_horizon = pred_horizon
        super().__init__(graphs_dir, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self): return ["monthly.pt"]

    def process(self):
        datas = []
        for month_dir in sorted(self.graphs_dir.glob("*")):
            nodes = pd.read_parquet(month_dir/"nodes.parquet")  # id, ticker, x_node (np array saved as list)
            edges = pd.read_parquet(month_dir/"edges.parquet")  # src, dst, etype_id, weight
            y = pd.read_parquet(month_dir/"labels.parquet")     # node_id, target (e.g., next 3-day return)

            x = torch.tensor(np.vstack(nodes["x_node"].to_list()), dtype=torch.float32)
            edge_index = torch.tensor(edges[["src","dst"]].values.T, dtype=torch.long)
            edge_weight = torch.tensor(edges["weight"].values, dtype=torch.float32)
            etype = torch.tensor(edges["etype_id"].values, dtype=torch.long)
            target = torch.tensor(y["target"].values, dtype=torch.float32)

            data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)
            data.etype = etype
            data.y = target
            data.node_id = torch.tensor(nodes["id"].values, dtype=torch.long)
            datas.append(data)

        data, slices = self.collate(datas)
        torch.save((data, slices), self.processed_paths[0])



# src/models/gnn_backbone.py
import torch, torch.nn as nn
from torch_geometric.nn import SAGEConv

class GraphSAGEBackbone(nn.Module):
    def __init__(self, in_dim: int, hid: int = 128, out_dim: int = 128, layers: int = 2, dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList()
        dims = [in_dim] + [hid]*(layers-1) + [out_dim]
        for i in range(len(dims)-1):
            self.convs.append(SAGEConv(dims[i], dims[i+1]))
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_weight=None):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs)-1:
                x = self.act(x)
                x = self.dropout(x)
        return x  # node embeddings [N, out_dim]


# src/models/seq_backbone.py
import torch.nn as nn

class SeqBackbone(nn.Module):
    def __init__(self, pretrained_baseline):
        super().__init__()
        self.base = pretrained_baseline  # load from repo checkpoint or class

    def forward(self, node_batch):
        """
        node_batch should carry per-node price/volume/sentiment sequences for the window
        Return: [N, d_seq]
        """
        return self.base(node_batch)  # ensure it returns embeddings not final logits


# src/models/fusion_head.py
import torch, torch.nn as nn

class FusionModel(nn.Module):
    def __init__(self, gnn: nn.Module, seq: nn.Module, d_gnn: int, d_seq: int, d_out: int = 1):
        super().__init__()
        self.gnn = gnn
        self.seq = seq
        self.proj = nn.Sequential(
            nn.Linear(d_gnn + d_seq, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, d_out),
        )

    def forward(self, data, node_seq_batch):
        z_g = self.gnn(data.x, data.edge_index, getattr(data, "edge_weight", None))   # [N, d_gnn]
        z_s = self.seq(node_seq_batch)                                                # [N, d_seq]
        z = torch.cat([z_g, z_s], dim=-1)
        return self.proj(z).squeeze(-1)                                              # [N]



# src/training/train_fusion.py
import torch, torch.nn as nn, torch.optim as optim
from torch_geometric.loader import NeighborLoader
from .metrics import r2_score

def train(model, loader, val_loader, epochs=50, lr=3e-4, wd=1e-4, device="cuda"):
    model.to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scaler = torch.cuda.amp.GradScaler()
    best = {"val_r2": -1e9, "state": None}

    for ep in range(epochs):
        model.train()
        for batch in loader:
            batch = batch.to(device)
            node_seq_batch = batch.node_seq.to(device)  # pre-prepared tensor per node
            with torch.cuda.amp.autocast():
                pred = model(batch, node_seq_batch)
                loss = nn.MSELoss()(pred, batch.y)
            scaler.scale(loss).step(opt); scaler.update(); opt.zero_grad(set_to_none=True)

        # val
        model.eval(); yhat=[]; ytrue=[]
        with torch.no_grad():
            for v in val_loader:
                v = v.to(device)
                node_seq = v.node_seq.to(device)
                p = model(v, node_seq)
                yhat.append(p.detach().cpu()); ytrue.append(v.y.cpu())
        yhat = torch.cat(yhat); ytrue = torch.cat(ytrue)
        r2 = r2_score(yhat, ytrue)
        if r2 > best["val_r2"]:
            best = {"val_r2": r2, "state": {k:v.cpu() for k,v in model.state_dict().items()}}
        print(f"epoch {ep} val R2={r2:.4f}")

    model.load_state_dict(best["state"])
    return model, best["val_r2"]
