"""
LLM-based Suspicious Activity Report (SAR) Generator

Uses GPT-4o-mini (via OpenAI API) or Llama-3-8B-Instruct (local HuggingFace)
to generate structured, compliance-ready SAR narratives from alert context.

Output format: JSON with fields:
  - narrative: str (EU-compliant SAR text)
  - risk_level: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
  - regulatory_flags: list[str]  (AMLD6, EBA, GDPR citations)
  - recommended_action: str
"""

import json
import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)


SAR_SYSTEM_PROMPT = """You are a financial compliance AI assistant specialised in EU anti-money laundering regulations.
Your task is to generate a Suspicious Activity Report (SAR) narrative based on transaction alert data.

Requirements:
1. Follow EU AMLD6 and EBA Guidelines on fraud reporting
2. Cite relevant regulatory frameworks (AMLD6, GDPR Art. 22, EU AI Act Art. 9-11)
3. Be factual, concise, and legally defensible
4. Use chain-of-thought reasoning: first analyse the evidence, then write the narrative
5. Output ONLY valid JSON matching the schema provided

JSON Schema:
{
  "narrative": "string (1-3 paragraphs, formal compliance language)",
  "risk_level": "LOW|MEDIUM|HIGH|CRITICAL",
  "regulatory_flags": ["list of relevant regulations"],
  "recommended_action": "string (freeze/investigate/escalate/monitor)",
  "confidence_score": float (0-1, based on model fraud scores),
  "evidence_summary": "string (1 sentence summarising key evidence)"
}"""


def build_alert_context(transaction: dict, fraud_scores: dict,
                        shap_values: dict = None) -> str:
    """Build the user prompt from alert data."""
    ctx = {
        "transaction": {
            "id": transaction.get("tx_id"),
            "amount": transaction.get("amount"),
            "sender": transaction.get("sender_id"),
            "receiver": transaction.get("receiver_id"),
            "merchant": transaction.get("merchant_id"),
            "timestamp": str(transaction.get("timestamp")),
            "type": transaction.get("tx_type", "PAYMENT"),
        },
        "fraud_scores": {
            "gat_score": round(fraud_scores.get("gat", 0), 4),
            "xgboost_score": round(fraud_scores.get("xgboost", 0), 4),
            "isolation_forest_score": round(fraud_scores.get("isolation_forest", 0), 4),
            "autoencoder_score": round(fraud_scores.get("autoencoder", 0), 4),
            "ensemble_score": round(fraud_scores.get("ensemble", 0), 4),
        },
        "velocity_features": {
            "tx_count_1h": transaction.get("velocity_1h", 0),
            "tx_count_24h": transaction.get("velocity_24h", 0),
        },
    }
    if shap_values:
        # Top-5 most influential features
        sorted_shap = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        ctx["top_shap_features"] = {k: round(v, 4) for k, v in sorted_shap}

    return f"Alert context:\n{json.dumps(ctx, indent=2)}\n\nGenerate the SAR report:"


class SARGenerator:

    def __init__(self, provider: str = "openai", model: str = "gpt-4o-mini",
                 local_model_path: str = None):
        self.provider = provider
        self.model = model
        self.local_model_path = local_model_path
        self._client = None
        self._local_pipeline = None

    def _get_openai_client(self):
        if self._client is None:
            from openai import OpenAI
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set. "
                                 "Add it to your .env file.")
            self._client = OpenAI(api_key=api_key)
        return self._client

    def _get_local_pipeline(self):
        if self._local_pipeline is None:
            from transformers import pipeline
            import torch
            log.info(f"Loading local LLM from {self.local_model_path}...")
            self._local_pipeline = pipeline(
                "text-generation",
                model=self.local_model_path,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        return self._local_pipeline

    def generate(self, transaction: dict, fraud_scores: dict,
                 shap_values: dict = None) -> dict:
        """
        Generate a SAR report for a flagged transaction.
        Returns parsed JSON dict.
        """
        user_prompt = build_alert_context(transaction, fraud_scores, shap_values)

        if self.provider == "openai":
            return self._generate_openai(user_prompt)
        else:
            return self._generate_local(user_prompt)

    def _generate_openai(self, user_prompt: str) -> dict:
        client = self._get_openai_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SAR_SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.2,        # low temperature for consistent, factual output
            response_format={"type": "json_object"},
            max_tokens=1024,
        )
        raw = response.choices[0].message.content
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            log.warning("Failed to parse LLM JSON output, returning raw text")
            return {"narrative": raw, "risk_level": "UNKNOWN",
                    "regulatory_flags": [], "recommended_action": "REVIEW"}

    def _generate_local(self, user_prompt: str) -> dict:
        pipe = self._get_local_pipeline()
        full_prompt = f"<|system|>\n{SAR_SYSTEM_PROMPT}\n<|user|>\n{user_prompt}\n<|assistant|>"
        result = pipe(full_prompt, max_new_tokens=512, temperature=0.2, do_sample=True)
        generated = result[0]["generated_text"].split("<|assistant|>")[-1].strip()
        try:
            # Find JSON block
            start = generated.find("{")
            end = generated.rfind("}") + 1
            return json.loads(generated[start:end])
        except Exception:
            return {"narrative": generated, "risk_level": "UNKNOWN",
                    "regulatory_flags": [], "recommended_action": "REVIEW"}
