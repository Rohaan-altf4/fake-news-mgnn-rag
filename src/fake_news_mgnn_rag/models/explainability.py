import os
import json
from typing import Dict, List, Any

# Attempt to import google genai
try:
    from google import genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

class LLMExplainer:
    """
    Generates human-readable explanations for the MGNN's predictions
    using the Gemini API, based on the graph attention weights and retrieved RAG facts.
    """
    def __init__(self, model_name: str = "gemini-2.5-flash", api_key: str = None):
        self.model_name = model_name
        
        # Priority: explicit api_key > GEMINI_API_KEY env var
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        
        if self.api_key and HAS_GENAI:
            self.client = genai.Client(api_key=self.api_key)
            self.use_mock = False
        else:
            print("WARNING: google-genai or GEMINI_API_KEY not found. Using mock explainability.")
            self.use_mock = True

    def generate_explanation(
        self,
        claim: str,
        predicted_prob: float,
        rag_facts: List[Dict[str, Any]],
        attention_summary: Dict[str, Any]
    ) -> str:
        """
        Generate the explanation using the provided context.
        
        Args:
            claim: The raw text of the social media post.
            predicted_prob: The model's predicted probability of being FAKE.
            rag_facts: List of facts retrieved from the RAG pipeline.
            attention_summary: A dictionary mapping modalities/facts to their 
                               relative attention weights (e.g., {'image': 0.8, 'fact_0': 0.1}).
        """
        prompt = self._build_prompt(claim, predicted_prob, rag_facts, attention_summary)
        
        if self.use_mock:
            return self._mock_generation(prompt, predicted_prob, attention_summary)
            
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            return response.text
        except Exception as e:
            return f"Error generating explanation via LLM: {str(e)}\n\nMock fallback:\n" + self._mock_generation(prompt, predicted_prob, attention_summary)

    def _build_prompt(self, claim: str, predicted_prob: float, rag_facts: List[Dict], attention_summary: Dict) -> str:
        prediction_label = "FAKE" if predicted_prob >= 0.5 else "REAL"
        confidence = max(predicted_prob, 1.0 - predicted_prob) * 100
        
        facts_str = ""
        for i, fact in enumerate(rag_facts):
            facts_str += f"- Fact {i}: {fact.get('content', '')}\n"
            
        att_str = json.dumps(attention_summary, indent=2)

        prompt = f"""You are an explainable AI system for detecting fake news.

The user posted the following claim:
"{claim}"

Our Multimodal Graph Neural Network (MGNN) analyzed this claim along with its associated image and retrieved ground-truth facts. 
It predicts with {confidence:.1f}% confidence that this post is {prediction_label}.

During its reasoning, the model focused its attention across different modalities. Here are the normalized attention weights:
{att_str}

Here are the retrieved ground-truth facts:
{facts_str}

Based on these attention weights and the facts, write a concise, human-readable paragraph explaining WHY the model made this prediction. 
If the model paid high attention to the image, it likely detected a cross-modal contradiction.
If the model paid high attention to a specific fact, it likely found the claim contradicting or aligning with that fact.

Output only the explanation.
"""
        return prompt

    def _mock_generation(self, prompt: str, prob: float, att_summary: Dict) -> str:
        label = "fake" if prob >= 0.5 else "real"
        highest_att = max(att_summary.items(), key=lambda x: x[1]) if att_summary else ("unknown", 0.0)
        
        return f"(Mocked Output) The system predicts this is {label} (prob: {prob:.2f}). The highest attention was placed on '{highest_att[0]}' (weight: {highest_att[1]:.2f}), indicating this component strongly influenced the decision."

if __name__ == "__main__":
    print("Testing LLMExplainer...")
    explainer = LLMExplainer()
    
    claim = "The Eiffel Tower was moved to London yesterday."
    prob = 0.95
    facts = [{"content": "The Eiffel Tower is located in Paris, France and has never been moved."}]
    att_summary = {"image": 0.1, "fact_0": 0.9}
    
    explanation = explainer.generate_explanation(claim, prob, facts, att_summary)
    print("\nGenerated Explanation:\n" + explanation)

