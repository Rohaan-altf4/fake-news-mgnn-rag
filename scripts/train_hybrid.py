import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

from fake_news_mgnn_rag.data_loaders.multimodal_dataset import get_multimodal_dataloader
from fake_news_mgnn_rag.encoders.text_encoder import TextEncoder
from fake_news_mgnn_rag.encoders.vision_encoder import VisionEncoder
from fake_news_mgnn_rag.rag.retriever import DynamicRAGRetriever
from fake_news_mgnn_rag.rag.vector_store import FactVectorStore
from fake_news_mgnn_rag.graphs.builder import GraphBuilder
from fake_news_mgnn_rag.models.hybrid_model import HybridFakeNewsModel

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Paths
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../"))

    val_json = os.path.join(REPO_ROOT, "data/raw/mmfakebench_raw/MMFakeBench_val.json")
    val_images = os.path.join(REPO_ROOT, "data/raw/mmfakebench_raw/images_val")

    test_json = os.path.join(REPO_ROOT, "data/raw/mmfakebench_raw/MMFakeBench_test.json")
    test_images = os.path.join(REPO_ROOT, "data/raw/mmfakebench_raw/images_test")

    # Hyperparameters
    batch_size = 4
    num_epochs = 2
    learning_rate = 2e-5

    # 1. Load Data
    print("Loading data...")
    train_loader = get_multimodal_dataloader(
        mmfakebench_json=val_json,
        mmfakebench_images=val_images,
        dataset_source="mmfakebench",
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    test_loader = get_multimodal_dataloader(
        mmfakebench_json=test_json,
        mmfakebench_images=test_images,
        dataset_source="mmfakebench",
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    # 2. Initialize Encoders and Retriever
    print("Initializing components...")
    text_encoder = TextEncoder().to(device)
    vision_encoder = VisionEncoder().to(device)

    vector_store = FactVectorStore(persist_directory=os.path.join(REPO_ROOT, "data/chroma_db"))
    retriever = DynamicRAGRetriever(vector_store=vector_store, top_k=2)

    builder = GraphBuilder(
        text_encoder=text_encoder,
        vision_encoder=vision_encoder,
        retriever=retriever,
        device=device
    )

    # 3. Initialize Model
    in_dims = {
        'text': text_encoder.hidden_size,
        'image': vision_encoder.hidden_size,
        'user': text_encoder.hidden_size,
        'rag_fact': text_encoder.hidden_size
    }
    model = HybridFakeNewsModel(
        in_dims=in_dims,
        hidden_dim=256,
        num_gnn_layers=2,
        num_heads=4,
        dropout=0.2,
        classifier_hidden_dim=128
    ).to(device)

    # 4. Optimizer and Loss
    # We only train the MGNN and Classifier head. Encoders are frozen by default if specified,
    # but here we just optimize the model parameters to save memory.
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    # 5. Training Loop
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        # Train loop
        train_pbar = tqdm(enumerate(train_loader), total=2, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for i, batch in train_pbar:
            optimizer.zero_grad()

            # Construct graph
            data = builder(batch)

            # Forward pass
            logits = model(data)
            labels = data['text'].y.float()

            # Loss
            loss = criterion(logits.squeeze(-1), labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Metrics
            probs = torch.sigmoid(logits.squeeze(-1))
            preds = (probs > 0.5).long()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            train_pbar.set_postfix({'loss': loss.item()})

            if i >= 1: # Break early for quick test
                break

        train_acc = accuracy_score(all_labels, all_preds)
        train_f1 = f1_score(all_labels, all_preds, zero_division=0)
        print(f"Epoch {epoch+1} Train - Loss: {total_loss/2:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            val_pbar = tqdm(enumerate(test_loader), total=2, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for i, batch in val_pbar:
                data = builder(batch)
                logits = model(data)
                labels = data['text'].y.float()

                loss = criterion(logits.squeeze(-1), labels)
                val_loss += loss.item()

                probs = torch.sigmoid(logits.squeeze(-1))
                preds = (probs > 0.5).long()
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

                val_pbar.set_postfix({'loss': loss.item()})

                if i >= 1:
                    break

        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, zero_division=0)
        print(f"Epoch {epoch+1} Val - Loss: {val_loss/2:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

    print("Training complete!")

if __name__ == "__main__":
    train()
