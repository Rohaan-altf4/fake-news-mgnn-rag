import torch
import torch.nn as nn
from typing import Dict, List, Optional
from typing import Dict, List, Optional, Tuple, Any
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATv2Conv, HeteroConv
from torch_geometric.nn import GATv2Conv
import torch.nn.functional as F

class FeatureProjection(nn.Module):
    """
    Projects node features of varying dimensions into a unified hidden space.
    Uses Linear -> LayerNorm -> ELU.
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.elu(self.norm(self.proj(x)))

class MGNNGATLayer(nn.Module):
    """
    A single Heterogeneous GAT layer using GATv2Conv to compute dynamic attention
    between different modalities (text, image, user, rag_fact).
    Rewritten manually (instead of HeteroConv) to allow extracting attention weights.
    """
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout

        # Out channels per head must be hidden_dim // num_heads to maintain hidden_dim after concatenation
        out_channels_per_head = hidden_dim // num_heads

        # Define edge types for message passing
        edge_types = [
        self.edge_types = [
            ('user', 'mentions', 'text'),
            ('text', 'mentioned_by', 'user'),
            ('text', 'consistent_with', 'image'),
            ('image', 'consistent_with', 'text'),
            ('text', 'grounded_in', 'rag_fact'),
            ('rag_fact', 'supports', 'text')
        ]

        # Create HeteroConv using GATv2Conv for every edge type
        conv_dict = {
            edge_type: GATv2Conv(
                in_channels=hidden_dim,
        # Create GATv2Conv for every edge type.
        # We use a ModuleDict where keys are string representations of edge types.
        self.convs = nn.ModuleDict({
            '__'.join(edge_type): GATv2Conv(
                in_channels=(hidden_dim, hidden_dim),
                out_channels=out_channels_per_head,
                heads=num_heads,
                concat=True,
                dropout=dropout,
                add_self_loops=False # HeteroConv typically handles self loops or we add residuals manually
            ) for edge_type in edge_types
        }
                add_self_loops=False
            ) for edge_type in self.edge_types
        })

        # We use sum aggregation for messages arriving from different edge types to the same node type
        self.conv = HeteroConv(conv_dict, aggr='sum')

        # Layer Norms for each node type after message passing
        self.norms = nn.ModuleDict({
            node_type: nn.LayerNorm(hidden_dim)
            for node_type in ['text', 'image', 'user', 'rag_fact']
        })

    def forward(self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict[tuple, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Apply heterogeneous convolution
        out_dict = self.conv(x_dict, edge_index_dict)
    def forward(
        self, 
        x_dict: Dict[str, torch.Tensor], 
        edge_index_dict: Dict[tuple, torch.Tensor],
        return_attention_weights: bool = False
    ):
        out_dict_list = {}
        attention_dict = {}

        # Apply residuals, layernorm, and dropout
        # 1. Message passing per edge type
        for edge_type, edge_index in edge_index_dict.items():
            src, rel, dst = edge_type
            edge_key = '__'.join(edge_type)
            
            if edge_key not in self.convs:
                continue

            conv = self.convs[edge_key]
            
            x_src = x_dict[src]
            x_dst = x_dict[dst]
            
            if return_attention_weights:
                out, alpha = conv((x_src, x_dst), edge_index, return_attention_weights=True)
                attention_dict[edge_type] = alpha
            else:
                out = conv((x_src, x_dst), edge_index)
                
            if dst not in out_dict_list:
                out_dict_list[dst] = []
            out_dict_list[dst].append(out)

        # 2. Aggregation (sum), Residuals, LayerNorm, and Dropout
        result_dict = {}
        for node_type, x in x_dict.items():
            if node_type in out_dict:
                out = out_dict[node_type]
            if node_type in out_dict_list:
                # Sum aggregation
                out = sum(out_dict_list[node_type])
                # Dropout
                out = F.dropout(out, p=self.dropout, training=self.training)
                # Residual connection
                out = out + x
                # Layer Norm + Activation
                out = F.elu(self.norms[node_type](out))
                result_dict[node_type] = out
            else:
                # If a node type receives no messages (e.g. isolated node), just return its input
                # If a node type receives no messages, just return its input
                result_dict[node_type] = x

        if return_attention_weights:
            return result_dict, attention_dict
        return result_dict

class MultimodalGNN(nn.Module):
    """
    The full Heterogeneous Graph Attention Network.
    Projects inputs to a shared space, then stacks N MGNNGATLayers.
    """
    def __init__(
        self,
        in_dims: Dict[str, int],
        hidden_dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.2
    ):
        super().__init__()

        # Projection layers for each node type
        self.projections = nn.ModuleDict({
            node_type: FeatureProjection(in_dim, hidden_dim)
            for node_type, in_dim in in_dims.items()
        })

        # Stacked GAT layers
        self.layers = nn.ModuleList([
            MGNNGATLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
    def forward(
        self, 
        data: HeteroData, 
        return_attention_weights: bool = False
    ):
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict

        # 1. Project all nodes to unified hidden space
        h_dict = {}
        for node_type, x in x_dict.items():
            if node_type in self.projections:
                h_dict[node_type] = self.projections[node_type](x)
            else:
                # Fallback if unconfigured node type is passed
                h_dict[node_type] = x

        # 2. Message Passing layers
        all_attentions = []
        for layer in self.layers:
            h_dict = layer(h_dict, edge_index_dict)
            if return_attention_weights:
                h_dict, att_dict = layer(h_dict, edge_index_dict, return_attention_weights=True)
                all_attentions.append(att_dict)
            else:
                h_dict = layer(h_dict, edge_index_dict, return_attention_weights=False)

        if return_attention_weights:
            return h_dict, all_attentions
        return h_dict

if __name__ == "__main__":
    # Self-contained verification block
    print("Running MGNN Verification Tests...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Instantiate Dummy HeteroData matching our schema
    data = HeteroData()

    num_nodes = 3
    # Text (1024d)
    data['text'].x = torch.randn(num_nodes, 1024)
    # Image (768d)
    data['image'].x = torch.randn(num_nodes, 768)
    # User (1024d)
    data['user'].x = torch.randn(num_nodes, 1024)
    # RAG Fact (1024d) - what if there are 0 facts for a batch? Let's test with some facts
    data['rag_fact'].x = torch.randn(num_nodes * 2, 1024)

    # Mock some edges
    # text <-> image (1 to 1)
    indices = torch.arange(num_nodes, dtype=torch.long)
    data['text', 'consistent_with', 'image'].edge_index = torch.stack([indices, indices], dim=0)
    data['image', 'consistent_with', 'text'].edge_index = torch.stack([indices, indices], dim=0)

    # user <-> text (1 to 1)
    data['user', 'mentions', 'text'].edge_index = torch.stack([indices, indices], dim=0)
    data['text', 'mentioned_by', 'user'].edge_index = torch.stack([indices, indices], dim=0)

    # text <-> rag_fact (1 text has 2 facts)
    src_text = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    dst_fact = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long)
    data['text', 'grounded_in', 'rag_fact'].edge_index = torch.stack([src_text, dst_fact], dim=0)
    data['rag_fact', 'supports', 'text'].edge_index = torch.stack([dst_fact, src_text], dim=0)

    data = data.to(device)

    # 2. Instantiate Model
    in_dims = {
        'text': 1024,
        'image': 768,
        'user': 1024,
        'rag_fact': 1024
    }
    hidden_dim = 256
    model = MultimodalGNN(in_dims=in_dims, hidden_dim=hidden_dim, num_layers=2, num_heads=4).to(device)

    print("Model instantiated successfully.")

    # Enable gradients on inputs to test backward pass
    for node_type in data.node_types:
        data[node_type].x.requires_grad_(True)

    # 3. Forward Pass
    out_dict = model(data)
    # 3. Forward Pass with Attention
    out_dict, attentions = model(data, return_attention_weights=True)

    print("\nForward pass outputs:")
    for k, v in out_dict.items():
        print(f"  {k}: shape {v.shape}")
        assert v.shape[1] == hidden_dim, f"Dimension mismatch for {k}: expected {hidden_dim}, got {v.shape[1]}"

    print("\nAttention extraction verified:")
    for layer_idx, att_dict in enumerate(attentions):
        print(f"  Layer {layer_idx}: {len(att_dict)} edge types with attention")
        for edge_type, (edge_index, alpha) in att_dict.items():
            print(f"    {edge_type}: alpha shape {alpha.shape}")

    print("\nDimensional alignment verified.")

    # 4. Backward Pass (dummy loss)
    # E.g. predicting binary classification from the text nodes
    dummy_labels = torch.randint(0, 2, (num_nodes,), dtype=torch.float, device=device)
    # Simple projection to 1 logit
    classifier = nn.Linear(hidden_dim, 1).to(device)

    logits = classifier(out_dict['text']).squeeze(-1)
    loss = F.binary_cross_entropy_with_logits(logits, dummy_labels)

    loss.backward()

    # Verify gradients flow back to the inputs
    for node_type in data.node_types:
        grad = data[node_type].x.grad
        assert grad is not None, f"No gradient for {node_type} input!"
        assert grad.abs().sum().item() > 0, f"Zero gradient for {node_type} input!"

    print("\nBackward pass & gradient flow verified end-to-end.")

