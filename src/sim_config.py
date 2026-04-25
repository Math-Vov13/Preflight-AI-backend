"""Centralized configuration, loaded from environment via .env."""
from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # SiliconFlow
    siliconflow_api_key: str = Field(..., alias="SILICONFLOW_API_KEY")
    siliconflow_base_url: str = Field(
        default="https://api.siliconflow.com/v1", alias="SILICONFLOW_BASE_URL"
    )

    # Zep Cloud (knowledge graph)
    zep_api_key: str = Field(default="", alias="ZEP_API_KEY")

    # Supabase auth — JWT signing secret used to validate Bearer tokens on
    # every protected route. When empty we run in *dev-local* mode: every
    # request is implicitly authenticated as `dev_user_id` so the rest of
    # the stack can boot without a Supabase project. Set the secret + the
    # public anon-key (frontend only) in `.env` to enable real auth.
    supabase_jwt_secret: str = Field(default="", alias="SUPABASE_JWT_SECRET")
    dev_user_id: str = Field(default="dev-local", alias="DEV_USER_ID")

    # Other external services
    tavily_api_key: str = Field(default="", alias="TAVILY_API_KEY")
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")

    # Task-based model bindings — each PreFlight service picks the model for
    # its own task. This replaces ai-agent-demo's tier-based Leader/Smart/Bulk.
    ontology_model: str = Field(
        default="Qwen/Qwen2.5-72B-Instruct", alias="ONTOLOGY_MODEL"
    )
    persona_model: str = Field(
        default="Qwen/Qwen2.5-72B-Instruct", alias="PERSONA_MODEL"
    )
    # Qwen2.5-7B kept choking on compound JSON; Qwen3-8B (also free tier) is
    # materially better at structured output.
    simulation_model: str = Field(
        default="Qwen/Qwen3-8B", alias="SIMULATION_MODEL"
    )
    report_model: str = Field(
        default="deepseek-ai/DeepSeek-R1", alias="REPORT_MODEL"
    )
    judge_model: str = Field(default="deepseek-ai/DeepSeek-V3", alias="JUDGE_MODEL")
    # Chat model — user-facing Q&A on ValidationReport. Fast conversational tier.
    chat_model: str = Field(default="deepseek-ai/DeepSeek-V3", alias="CHAT_MODEL")
    embedding_model: str = Field(
        default="Qwen/Qwen3-Embedding-0.6B", alias="EMBEDDING_MODEL"
    )

    budget_usd: float = Field(default=50.0, alias="BUDGET_USD")


_settings: Settings | None = None


def settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()  # type: ignore[call-arg]
    return _settings
