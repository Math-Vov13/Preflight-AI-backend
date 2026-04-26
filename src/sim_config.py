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
    # Kimi K2 — non-reasoning instruct. We had Qwen3-8B before; ~94% of its
    # output tokens were "reasoning" the persona never actually said, which
    # tripled per-call latency for no UX gain (forum chatter doesn't need
    # internal monologue). K2 writes the post directly, so a 10-persona /
    # 3-round run goes from ~3.5 min in this phase to ~70-90s. Override
    # via SIMULATION_MODEL=... in .env if SiliconFlow renames the revision.
    simulation_model: str = Field(
        default="moonshotai/Kimi-K2.6", alias="SIMULATION_MODEL"
    )
    report_model: str = Field(
        default="deepseek-ai/DeepSeek-R1", alias="REPORT_MODEL"
    )
    judge_model: str = Field(default="deepseek-ai/DeepSeek-V3", alias="JUDGE_MODEL")
    # Chat model — user-facing Q&A on ValidationReport. Fast conversational tier.
    chat_model: str = Field(default="deepseek-ai/DeepSeek-V3", alias="CHAT_MODEL")
    # Model used by the orchestrator passthrough (`/generation/` with
    # `tools_passthrough=True`). MUST support OpenAI-compatible tool calling
    # cleanly. Default `Qwen/Qwen2.5-72B-Instruct` is known good on
    # SiliconFlow's adapter; DeepSeek-V3 here is BROKEN — its native
    # `<｜tool▁call▁begin｜>` tokens leak into `function.arguments` and the
    # parsed args come back empty. Keep this distinct from `chat_model`.
    orchestrator_model: str = Field(
        default="Qwen/Qwen2.5-72B-Instruct", alias="ORCHESTRATOR_MODEL",
    )
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
