import torch
from fake_news_mgnn_rag.encoders.text_encoder import TextEncoder
from fake_news_mgnn_rag.encoders.vision_encoder import VisionEncoder
from fake_news_mgnn_rag.rag.retriever import DynamicRAGRetriever
from fake_news_mgnn_rag.graphs.builder import GraphBuilder
from fake_news_mgnn_rag.models.hybrid_model import HybridFakeNewsModel

def test_pipeline():
    print("Initializing components...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Use real encoders for testing
    text_encoder = TextEncoder().to(device)
    vision_encoder = VisionEncoder().to(device)

    # We can mock the retriever or point it to a test chromadb
    # But for a quick test we can just mock the retrieve_batch method
    # to avoid needing a populated DB if it's empty.
    retriever = DynamicRAGRetriever(vector_store=None)

    # Let's mock the RAG retriever to return 1 fact per claim
    def mock_retrieve_batch(queries, top_k):
        return [[{"content": f"Mock fact for claim: {q}", "timestamp": "2023-01-01"}] for q in queries]
    retriever.retrieve_batch = mock_retrieve_batch

    builder = GraphBuilder(
        text_encoder=text_encoder,
        vision_encoder=vision_encoder,
        retriever=retriever,
        device=device
    )

    in_dims = {
        'text': text_encoder.hidden_size,
        'image': vision_encoder.hidden_size,
        'user': text_encoder.hidden_size,
        'rag_fact': text_encoder.hidden_size
    }

    model = HybridFakeNewsModel(in_dims=in_dims).to(device)

    print("Components initialized. Constructing batch...")

    # Dummy batch
    batch = {
        "text": ["The moon is made of green cheese.", "Aliens landed in Roswell."],
        "image": torch.randn(2, 3, 224, 224), # Dummy images for ViT
        "dataset_origin": ["test_source", "test_source_2"],
        "label": torch.tensor([1, 0])
    }

    print("Building HeteroData graph...")
    data = builder(batch)
    print("Graph node types:", data.node_types)
    print("Graph edge types:", data.edge_types)

    print("Running forward pass...")
    logits, attentions = model(data, return_attention_weights=True)

    print("Logits:", logits)

    print("Running explainability for a single item...")
    single_batch = {
        "text": ["The moon is made of green cheese."],
        "image": torch.randn(1, 3, 224, 224),
        "dataset_origin": ["test_source"],
        "label": torch.tensor([1])
    }
    single_data = builder(single_batch)

    mock_rag_facts = [{"content": fact["content"]} for fact in retriever.retrieve_batch([single_batch["text"][0]], top_k=1)[0]]
    prob, explanation = model.predict_and_explain(
        data=single_data,
        claim=single_batch["text"][0],
        rag_facts=mock_rag_facts
    )

    print("\nExplanation for single item:")
    print("Prob:", prob)
    print(explanation)
    print("\nPipeline test complete!")

if __name__ == "__main__":
    test_pipeline()
