import torch
import torch.nn as nn
import torch.nn.functional as F

class FakeNewsClassifierHead(nn.Module):
    """
    Final Multi-Layer Perceptron (MLP) for predicting Fake vs Real.
    Takes the aggregated text node embedding from the MGNN and outputs a logit.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_embeddings: Shape (num_claims, input_dim)
        Returns:
            logits: Unnormalized predictions (num_claims, 1)
        """
        x = self.fc1(text_embeddings)
        x = self.norm(x)
        x = F.elu(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits

    def predict_proba(self, text_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Returns probabilities (Fake = 1, Real = 0).
        """
        logits = self.forward(text_embeddings)
        return torch.sigmoid(logits)

if __name__ == "__main__":
    print("Testing FakeNewsClassifierHead...")
    dummy_input = torch.randn(4, 256)
    model = FakeNewsClassifierHead(input_dim=256, hidden_dim=128)
    
    logits = model(dummy_input)
    probs = model.predict_proba(dummy_input)
    
    print(f"Logits shape: {logits.shape}")
    print(f"Probs shape: {probs.shape}")
    assert logits.shape == (4, 1)
    assert probs.shape == (4, 1)
    print("Classifier Head tests passed!")

