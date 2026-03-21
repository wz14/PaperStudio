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

    # LLM 自定义端点（兼容 OpenAI 协议的第三方或本地服务）
    # 三项均设置后，rag_service 会通过 llm_config 传入 PaperQA。
    llm_base_url: str = Field(default="", validation_alias="LLM_BASE_URL")
    llm_api_key: str = Field(default="", validation_alias="LLM_API_KEY")
    llm_model: str = Field(default="", validation_alias="LLM_MODEL")

    # PaperQA / LiteLLM（OPENAI_API_KEY 等由 litellm 读取）
    # llm_model 未设置时，以下三项作为默认模型名称。
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

    # Web UI 认证（格式："user1:pass1,user2:pass2"）
    web_users: str = Field(default="admin:paperstudio", validation_alias="WEB_USERS")
    # JWT 签名密钥，生产环境请务必替换为随机长字符串
    web_secret_key: str = Field(
        default="paperstudio-dev-key-change-in-production",
        validation_alias="WEB_SECRET_KEY",
    )
