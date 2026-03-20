"""微信消息业务：命令解析、文献入库、PaperQA 异步回复。"""

from __future__ import annotations

import logging
import re

from app.config import AppConfig
from app.db import append_message, get_session_factory, upsert_user
from app import rag_service
from app.wechat import send_custom_text

logger = logging.getLogger(__name__)

HELP_TEXT = """科研助手（PaperQA）命令：
• 直接输入问题：基于你已上传的 PDF/文献 作答
• 添加 <论文URL>：下载并加入个人文献库（支持 arXiv、DOI 链接、.pdf 等）
• 发送「链接」消息或消息里带论文 URL：自动尝试下载
• 发送图片/文件：图片会保存为素材（复杂 PDF 请用链接或「添加」）
• 文献：查看当前文献库文件名
• 帮助：显示本说明

说明：服务号被动回复需在数秒内完成，长回答通过「客服消息」发送，请稍候。"""


def _strip_cmd(s: str) -> str:
    return s.strip()


async def handle_text_message(
    cfg: AppConfig, openid: str, content: str
) -> tuple[str, str | None]:
    """(同步回复正文, 可选：需异步 PaperQA 的问题). 同步正文为空串时表示仅走异步。"""
    text = _strip_cmd(content)
    if not text:
        return "请输入内容。", None

    if text in ("帮助", "help", "?", "？"):
        return HELP_TEXT, None

    if text in ("文献", "列表", "papers", "我的文献"):
        papers = await rag_service.list_papers(cfg, openid)
        if not papers:
            return "文献库为空。请发送「添加 https://arxiv.org/pdf/...」或带论文链接的消息。", None
        body = "\n".join(papers[:50])
        more = f"\n… 共 {len(papers)} 个" if len(papers) > 50 else ""
        return f"你的文献库：\n{body}{more}", None

    m_add = re.match(r"^添加\s+(\S+)", text, re.IGNORECASE)
    if m_add:
        url = m_add.group(1)
        return await rag_service.ingest_url(cfg, openid, url), None

    maybe = await rag_service.maybe_ingest_first_url(cfg, openid, text)
    if maybe:
        return maybe, None

    # 默认走问答
    return "", text


async def run_rag_background(
    cfg: AppConfig, openid: str, question: str, *, app_id: str, app_secret: str
) -> None:
    """客服消息：长回答 + 记录对话。"""
    factory = get_session_factory()
    async with factory() as session:
        await upsert_user(session, openid)
        await append_message(session, openid, "user", question)

    # ask_paperqa 内含 DirectAnswer 工具，无文献时 agent 会自动直接回答
    try:
        answer = await rag_service.ask_paperqa(cfg, openid, question)
    except Exception as e:
        logger.exception("ask_paperqa 失败 openid=%s", openid[:8])
        err = f"处理失败：{e!s}"
        async with factory() as session:
            await append_message(session, openid, "assistant", err)
        await send_custom_text(app_id, app_secret, openid, err)
        return

    async with factory() as session:
        await append_message(session, openid, "assistant", answer)

    await send_custom_text(app_id, app_secret, openid, answer)


async def run_link_background(
    cfg: AppConfig, openid: str, url: str, *, app_id: str, app_secret: str
) -> None:
    try:
        msg = await rag_service.ingest_url(cfg, openid, url)
        await send_custom_text(app_id, app_secret, openid, msg)
    except Exception as e:
        logger.exception("链接入库失败")
        await send_custom_text(app_id, app_secret, openid, f"链接处理失败：{e!s}")


async def run_media_background(
    cfg: AppConfig,
    openid: str,
    data: bytes,
    suffix: str,
    *,
    app_id: str,
    app_secret: str,
) -> None:
    try:
        msg = await rag_service.save_uploaded_bytes(cfg, openid, data, suffix)
        await send_custom_text(app_id, app_secret, openid, msg)
    except Exception as e:
        logger.exception("素材保存失败")
        await send_custom_text(app_id, app_secret, openid, f"保存失败：{e!s}")
