"""按微信用户隔离的 PaperQA（[Future-House/paper-qa](https://github.com/Future-House/paper-qa)）封装。"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from pathlib import Path

import httpx
from paperqa.agents.main import agent_query
from paperqa.settings import AgentSettings, IndexSettings, Settings

from app.config import AppConfig

logger = logging.getLogger(__name__)

# 简单 URL 检测（论文 PDF / arXiv 等）
_URL_RE = re.compile(
    r"https?://[^\s<>\"{}|\\^`\[\]]+",
    re.IGNORECASE,
)


def user_dir_hash(openid: str) -> str:
    """文件系统安全、不可逆的用户目录名。"""
    return hashlib.sha256(openid.encode("utf-8")).hexdigest()[:32]


def user_paths(cfg: AppConfig, openid: str) -> tuple[Path, Path]:
    h = user_dir_hash(openid)
    base = cfg.data_dir / "users" / h
    papers = base / "papers"
    indexes = base / "indexes"
    return papers, indexes


def _build_llm_config(cfg: AppConfig) -> dict | None:
    """若配置了自定义 LLM 端点，构造 litellm model_list 格式的 llm_config。"""
    if not (cfg.llm_base_url and cfg.llm_api_key and cfg.llm_model):
        return None
    return {
        "model_list": [
            {
                "model_name": cfg.llm_model,
                "litellm_params": {
                    "model": cfg.llm_model,
                    "api_base": cfg.llm_base_url,
                    "api_key": cfg.llm_api_key,
                },
            }
        ]
    }


def build_settings_for_user(cfg: AppConfig, openid: str) -> Settings:
    papers, indexes = user_paths(cfg, openid)
    papers.mkdir(parents=True, exist_ok=True)
    indexes.mkdir(parents=True, exist_ok=True)

    # 使用自定义模型名或 PAPERQA_LLM 环境变量
    llm_name = cfg.llm_model or cfg.paperqa_llm
    summary_llm_name = cfg.llm_model or cfg.paperqa_summary_llm
    llm_config = _build_llm_config(cfg)

    logger.info(
        "构建 PaperQA Settings: llm=%s summary_llm=%s custom_endpoint=%s",
        llm_name,
        summary_llm_name,
        bool(llm_config),
    )

    kwargs: dict = dict(
        llm=llm_name,
        summary_llm=summary_llm_name,
        embedding=cfg.paperqa_embedding,
        verbosity=0,
        agent=AgentSettings(
            timeout=cfg.paperqa_agent_timeout,
            index=IndexSettings(
                paper_directory=papers,
                index_directory=indexes,
                use_absolute_paper_directory=True,
                sync_with_paper_directory=True,
                recurse_subdirectories=True,
            ),
            rebuild_index=True,
        ),
    )
    if llm_config is not None:
        kwargs["llm_config"] = llm_config
        kwargs["summary_llm_config"] = llm_config

    return Settings(**kwargs)


async def list_papers(cfg: AppConfig, openid: str) -> list[str]:
    papers, _ = user_paths(cfg, openid)
    if not papers.exists():
        return []
    return sorted(
        f.name for f in papers.iterdir() if f.is_file() and not f.name.startswith(".")
    )


async def ingest_url(cfg: AppConfig, openid: str, url: str) -> str:
    """下载 URL 到用户 papers 目录，下次 ask 会同步进索引。"""
    papers, _ = user_paths(cfg, openid)
    papers.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(
        follow_redirects=True, timeout=120.0, headers={"User-Agent": "PaperStudio/1.0"}
    ) as client:
        r = await client.get(url)
        r.raise_for_status()
        body = r.content
        ct = r.headers.get("content-type", "").lower()

    # 从 URL 或 Content-Type 猜扩展名
    suffix = ".pdf"
    if "pdf" in ct:
        suffix = ".pdf"
    elif "html" in ct:
        suffix = ".html"
    elif "text" in ct:
        suffix = ".txt"

    name = hashlib.sha256(url.encode()).hexdigest()[:16] + suffix
    dest = papers / name
    dest.write_bytes(body)
    logger.info(
        "已保存文献: openid_hash=%s path=%s bytes=%d",
        user_dir_hash(openid),
        dest,
        len(body),
    )
    return f"已保存到文献库: {name}（正在建立索引，可直接提问）"


async def ask_paperqa(cfg: AppConfig, openid: str, question: str) -> str:
    """对用户文献库执行 PaperQA agent 问答。"""
    settings = build_settings_for_user(cfg, openid)
    logger.info("PaperQA 提问: openid_hash=%s q=%s...", user_dir_hash(openid), question[:80])
    try:
        response = await agent_query(question, settings)
    except Exception:
        logger.exception("PaperQA 执行失败 openid_hash=%s", user_dir_hash(openid))
        raise
    ans = response.session.answer.strip()
    if not ans:
        logger.error("PaperQA 返回空答案 status=%s", response.status)
        raise RuntimeError("模型未返回有效答案")
    return ans


def extract_urls(text: str) -> list[str]:
    return _URL_RE.findall(text)


async def save_uploaded_bytes(
    cfg: AppConfig, openid: str, data: bytes, suffix: str
) -> str:
    """将二进制保存为用户文献目录下的文件，供 PaperQA 索引。"""
    papers, _ = user_paths(cfg, openid)
    papers.mkdir(parents=True, exist_ok=True)
    name = hashlib.sha256(data).hexdigest()[:16] + suffix
    dest = papers / name
    dest.write_bytes(data)
    logger.info(
        "已写入用户文献: openid_hash=%s path=%s bytes=%d",
        user_dir_hash(openid),
        dest,
        len(data),
    )
    return f"已保存到文献库: {name}（正在建立索引，可直接提问）"


async def maybe_ingest_first_url(cfg: AppConfig, openid: str, text: str) -> str | None:
    """若文本含 URL，尝试拉取并入库（返回提示语）；否则 None。"""
    urls = extract_urls(text)
    if not urls:
        return None
    url = urls[0]
    # 仅对看起来像文献的链接自动拉取，避免误抓普通网页
    lower = url.lower()
    if not any(
        x in lower
        for x in (
            ".pdf",
            "arxiv",
            "doi.org",
            "ieee",
            "springer",
            "nature.com",
            "sciencedirect",
            "pmc",
            "ncbi",
            "biorxiv",
            "medrxiv",
        )
    ):
        logger.info("跳过非文献向 URL: %s", url[:80])
        return None
    return await ingest_url(cfg, openid, url)
