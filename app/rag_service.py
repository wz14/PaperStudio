"""按微信用户隔离的 PaperQA（[Future-House/paper-qa](https://github.com/Future-House/paper-qa)）封装。

问答路由策略：
- 文献库为空时直接调 LLM，跳过 agent
- 文献库有内容时走 PaperQA agent（注入了 direct_answer 工具）
- agent 返回 CANNOT_ANSWER_PHRASE（无相关证据）时也改用直接 LLM 回答
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

import httpx
import litellm
from aviary.core import Tool
from paperqa.agents.env import PaperQAEnvironment
from paperqa.agents.main import agent_query
from paperqa.agents.tools import EnvironmentState, NamedTool
from paperqa.prompts import CANNOT_ANSWER_PHRASE, env_reset_prompt
from paperqa.settings import AgentSettings, IndexSettings, Settings

from app.config import AppConfig

logger = logging.getLogger(__name__)

# 简单 URL 检测（论文 PDF / arXiv 等）
_URL_RE = re.compile(
    r"https?://[^\s<>\"{}|\\^`\[\]]+",
    re.IGNORECASE,
)

# agent_prompt 里追加的决策规则，强制指定何时调 direct_answer
_DIRECT_ANSWER_HINT = (
    "\n\n## Tool selection rules (MUST follow strictly)\n"
    "Before every action, check the current status:\n"
    "1. If Paper Count = 0, OR the question is casual/conversational/general knowledge:"
    " call **direct_answer** immediately. Do NOT call gen_answer or paper_search first.\n"
    "2. If you searched but Relevant Papers = 0 and Current Evidence = 0:"
    " call **direct_answer**. Do NOT call gen_answer.\n"
    "3. Only call gen_answer when you have gathered real evidence from papers.\n"
    "Violating these rules wastes tokens and produces useless answers."
)

# DirectAnswer 工具使用的系统提示词，鼓励有趣、自然的回答风格
_DIRECT_ANSWER_SYSTEM_PROMPT = (
    "You are PaperStudio, a sharp and personable research assistant who speaks both"
    " English and Chinese fluently."
    " You love discussing everything from cutting-edge research to everyday curiosities."
    " Be warm, witty, and natural — like a knowledgeable friend, not a textbook."
    " For casual or conversational questions, be playful and direct."
    " For academic questions without papers, share your knowledge confidently and clearly."
    " Keep answers focused: avoid unnecessary filler but don't be curt."
    " Always respond in the same language the user used."
)


class DirectAnswer(NamedTool):
    """向 PaperQA agent 注入的直接回答工具，无需搜索论文。"""

    TOOL_FN_NAME = "direct_answer"

    llm_model_name: str
    api_base: str = ""
    api_key: str = ""

    async def direct_answer(self, state: EnvironmentState) -> str:
        """
        Answer the question directly from AI knowledge, without searching papers.

        Use this tool when:
        - The question is conversational or general knowledge (not requiring academic papers).
        - Paper searches returned no results or no relevant papers were found.

        Args:
            state: Current state.

        Returns:
            The answer text followed by the current status string.
        """
        kwargs: dict = {
            "model": self.llm_model_name,
            "messages": [
                {"role": "system", "content": _DIRECT_ANSWER_SYSTEM_PROMPT},
                {"role": "user", "content": state.session.question},
            ],
        }
        if self.api_base and self.api_key:
            kwargs["api_base"] = self.api_base
            kwargs["api_key"] = self.api_key

        logger.info(
            "direct_answer 调用 model=%s q=%s...",
            self.llm_model_name,
            state.session.question[:60],
        )
        response = await litellm.acompletion(**kwargs)
        answer: str = response.choices[0].message.content or ""
        if not answer:
            raise RuntimeError("direct_answer: LLM 返回空答案")

        # 写入 session，供 complete 工具读取
        state.session.answer = answer.strip()
        return f"{answer} | {state.status}"


def _make_extended_env_class(cfg: AppConfig) -> type[PaperQAEnvironment]:
    """返回一个注入了 DirectAnswer 工具的 PaperQAEnvironment 子类。"""
    raw_model = cfg.llm_model or cfg.paperqa_llm
    # 自定义端点时加 openai/ 前缀；使用内置模型名（如 gpt-4o-mini）时不需要
    litellm_model = f"openai/{raw_model}" if cfg.llm_base_url else raw_model
    direct_tool_instance = DirectAnswer(
        llm_model_name=litellm_model,
        api_base=cfg.llm_base_url,
        api_key=cfg.llm_api_key,
    )

    class _ExtendedEnv(PaperQAEnvironment):
        def make_tools(self) -> list[Tool]:
            tools = super().make_tools()
            # 插入在 complete（末位）之前，使 agent 能看到并选择该工具
            dt = Tool.from_function(direct_tool_instance.direct_answer)
            tools.insert(-1, dt)
            return tools

    return _ExtendedEnv


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
    """若配置了自定义 LLM 端点，构造 litellm model_list 格式的 llm_config。

    litellm 要求 litellm_params.model 带 provider 前缀（如 openai/xxx）才能
    识别协议类型；model_name 保持原始名用于内部路由引用。
    """
    if not (cfg.llm_base_url and cfg.llm_api_key and cfg.llm_model):
        return None
    # 自定义端点均走 OpenAI 兼容协议，加 openai/ 前缀告知 litellm
    litellm_model = f"openai/{cfg.llm_model}"
    return {
        "model_list": [
            {
                "model_name": cfg.llm_model,
                "litellm_params": {
                    "model": litellm_model,
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

    llm_name = cfg.llm_model or cfg.paperqa_llm
    summary_llm_name = cfg.llm_model or cfg.paperqa_summary_llm
    llm_config = _build_llm_config(cfg)

    logger.info(
        "构建 PaperQA Settings: llm=%s summary_llm=%s custom_endpoint=%s",
        llm_name,
        summary_llm_name,
        bool(llm_config),
    )

    agent_kwargs: dict = dict(
        # 用同一模型做工具选择，保持一致
        agent_llm=llm_name,
        timeout=cfg.paperqa_agent_timeout,
        # 追加 direct_answer 的使用提示
        agent_prompt=env_reset_prompt + _DIRECT_ANSWER_HINT,
        index=IndexSettings(
            paper_directory=papers,
            index_directory=indexes,
            use_absolute_paper_directory=True,
            sync_with_paper_directory=True,
            recurse_subdirectories=True,
        ),
        rebuild_index=True,
    )
    if llm_config is not None:
        # agent 的 LLM 也需要走自定义端点
        agent_kwargs["agent_llm_config"] = llm_config

    kwargs: dict = dict(
        llm=llm_name,
        summary_llm=summary_llm_name,
        embedding=cfg.paperqa_embedding,
        verbosity=0,
        agent=AgentSettings(**agent_kwargs),
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


async def _direct_llm_answer(cfg: AppConfig, question: str) -> str:
    """跳过 PaperQA agent，直接调 LLM 回答。用于文献库为空或无相关证据的场景。"""
    raw_model = cfg.llm_model or cfg.paperqa_llm
    litellm_model = f"openai/{raw_model}" if cfg.llm_base_url else raw_model
    kwargs: dict = {
        "model": litellm_model,
        "messages": [
            {"role": "system", "content": _DIRECT_ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
    }
    if cfg.llm_base_url and cfg.llm_api_key:
        kwargs["api_base"] = cfg.llm_base_url
        kwargs["api_key"] = cfg.llm_api_key

    logger.info("直接 LLM 回答: model=%s q=%s...", litellm_model, question[:60])
    response = await litellm.acompletion(**kwargs)
    answer: str = response.choices[0].message.content or ""
    if not answer:
        raise RuntimeError("_direct_llm_answer: LLM 返回空答案")
    return answer.strip()


async def ask_paperqa(cfg: AppConfig, openid: str, question: str) -> str:
    """对用户文献库执行问答。

    路由策略：
    - 文献库为空 → 直接调 LLM，跳过 agent（省时省 token）
    - 文献库有内容 → 走 PaperQA agent（含 direct_answer 工具）
    - agent 返回"无法回答"哨兵短语 → 改用直接 LLM 回答
    """
    papers, _ = user_paths(cfg, openid)
    paper_count = (
        sum(1 for f in papers.iterdir() if f.is_file() and not f.name.startswith("."))
        if papers.exists()
        else 0
    )
    logger.info(
        "PaperQA 提问: openid_hash=%s paper_count=%d q=%s...",
        user_dir_hash(openid),
        paper_count,
        question[:80],
    )

    if paper_count == 0:
        logger.info("文献库为空，跳过 agent 直接调 LLM")
        return await _direct_llm_answer(cfg, question)

    settings = build_settings_for_user(cfg, openid)
    env_class = _make_extended_env_class(cfg)
    try:
        response = await agent_query(question, settings, env_class=env_class)
    except Exception:
        logger.exception("PaperQA 执行失败 openid_hash=%s", user_dir_hash(openid))
        raise

    ans = response.session.answer.strip()
    if not ans:
        logger.error("PaperQA 返回空答案 status=%s", response.status)
        raise RuntimeError("模型未返回有效答案")

    # agent 找到论文但无相关证据时会返回 CANNOT_ANSWER_PHRASE，改走直接 LLM
    if CANNOT_ANSWER_PHRASE in ans:
        logger.info("agent 无相关证据，改用直接 LLM 回答 openid_hash=%s", user_dir_hash(openid))
        return await _direct_llm_answer(cfg, question)

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
