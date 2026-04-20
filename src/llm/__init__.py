"""
LLM-based SAR (Suspicious Activity Report) generator.

Uses GPT-4o-mini (OpenAI API) or a local Llama-3-8B-Instruct model to
produce EU AMLD6-compliant SAR narratives from fraud alert context.

Requires: OPENAI_API_KEY environment variable for the OpenAI provider.
"""

from src.llm.sar_generator import SARGenerator

__all__ = ["SARGenerator"]
