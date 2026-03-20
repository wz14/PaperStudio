"""应用配置：环境变量与 PaperQA / 微信相关项。"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = Field(default="0.0.0.0", validation_alias="HOST")
    port: int = Field(default=8000, validation_alias="PORT")

    database_url: str = Field(
        default="sqlite+aiosqlite:///./data/paperstudio.db",
        validation_alias="DATABASE_URL",
    )
    data_dir: Path = Field(default=Path("./data"), validation_alias="DATA_DIR")

    # PaperQA / LiteLLM（OPENAI_API_KEY 等由 litellm 读取）
    paperqa_llm: str = Field(default="gpt-4o-mini", validation_alias="PAPERQA_LLM")
    paperqa_summary_llm: str = Field(
        default="gpt-4o-mini", validation_alias="PAPERQA_SUMMARY_LLM"
    )
    paperqa_embedding: str = Field(
        default="text-embedding-3-small", validation_alias="PAPERQA_EMBEDDING"
    )
    paperqa_agent_timeout: float = Field(
        default=120.0, validation_alias="PAPERQA_AGENT_TIMEOUT"
    )

    # 微信公众平台
    wechat_token: str = Field(default="", validation_alias="WECHAT_TOKEN")
    wechat_app_id: str = Field(default="", validation_alias="WECHAT_APP_ID")
    wechat_app_secret: str = Field(default="", validation_alias="WECHAT_APP_SECRET")
