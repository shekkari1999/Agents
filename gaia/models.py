"""GAIA benchmark data models."""

from pydantic import BaseModel


class GaiaOutput(BaseModel):
    """Structured output format for GAIA problem solutions."""
    is_solvable: bool
    unsolvable_reason: str = ""
    final_answer: str = ""

