import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, List
from torch_geometric.data import HeteroData

from fake_news_mgnn_rag.models.mgnn import MultimodalGNN
from fake_news_mgnn_rag.models.classifier import FakeNewsClassifierHead
from fake_news_mgnn_rag.models.explainability import LLMExplainer

class HybridFakeNewsModel(nn.Module):
    """
    The full Phase 2/3 Hybrid Model:
    MultimodalGNN -> Classifier Head -> Explainability
    """
    def __init__(
        self,
        in_dims: Dict[str, int],
        hidden_dim: int = 512,
        num_gnn_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.2,
        classifier_hidden_dim: int = 128
    ):
        super().__init__()
        self.mgnn = MultimodalGNN(
            in_dims=in_dims,
            hidden_dim=hidden_dim,
            num_layers=num_gnn_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.classifier = FakeNewsClassifierHead(
            input_dim=hidden_dim,
            hidden_dim=classifier_hidden_dim,
            dropout=dropout
        )
        
        self.explainer = LLMExplainer()

    def forward(self, data: HeteroData, return_attention_weights: bool = False):
        """
        Forward pass.
        Returns:
            logits (Tensor): (num_claims, 1)
            attention_dict (list of dicts, optional)
        """
        if return_attention_weights:
            out_dict, attentions = self.mgnn(data, return_attention_weights=True)
            text_embeddings = out_dict['text']
            logits = self.classifier(text_embeddings)
            return logits, attentions
        else:
            out_dict = self.mgnn(data, return_attention_weights=False)
            text_embeddings = out_dict['text']
            logits = self.classifier(text_embeddings)
            return logits

    @torch.no_grad()
    def predict_and_explain(
        self, 
        data: HeteroData, 
        claim: str, 
        rag_facts: List[Dict[str, Any]]
    ) -> Tuple[float, str]:
        """
        End-to-end inference for a SINGLE sample.
        Predicts the probability and generates the explanation.
        """
        self.eval()
        logits, attentions = self.forward(data, return_attention_weights=True)
        prob = torch.sigmoid(logits).item()
        
        # Summarize attention from the last GNN layer
        last_layer_att = attentions[-1]
        
        # Naive summarization: average the alpha weights for edges connected to the text node
        att_summary = {}
        for edge_type, (edge_index, alpha) in last_layer_att.items():
            if 'text' in edge_type:
                att_summary['__'.join(edge_type)] = float(alpha.mean().item())
                
        explanation = self.explainer.generate_explanation(
            claim=claim,
            predicted_prob=prob,
            rag_facts=rag_facts,
            attention_summary=att_summary
        )
        
        return prob, explanation

if __name__ == "__main__":
    print("Testing HybridFakeNewsModel...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    in_dims = {'text': 1024, 'image': 768, 'user': 1024, 'rag_fact': 1024}
    model = HybridFakeNewsModel(in_dims=in_dims, hidden_dim=256, classifier_hidden_dim=128).to(device)
    
    data = HeteroData()
    data['text'].x = torch.randn(1, 1024)
    data['image'].x = torch.randn(1, 768)
    data['user'].x = torch.randn(1, 1024)
    data['rag_fact'].x = torch.randn(2, 1024)
    
    idx = torch.tensor([0])
    data['text', 'consistent_with', 'image'].edge_index = torch.stack([idx, idx], dim=0)
    data['image', 'consistent_with', 'text'].edge_index = torch.stack([idx, idx], dim=0)
    
    src = torch.tensor([0, 0])
    dst = torch.tensor([0, 1])
    data['text', 'grounded_in', 'rag_fact'].edge_index = torch.stack([src, dst], dim=0)
    data['rag_fact', 'supports', 'text'].edge_index = torch.stack([dst, src], dim=0)
    
    data = data.to(device)
    
    prob, explanation = model.predict_and_explain(
        data=data,
        claim="The sky is green today.",
        rag_facts=[{"content": "The sky is blue."}]
    )
    
    print(f"\nProb: {prob:.4f}")
    print(f"Explanation:\n{explanation}")

