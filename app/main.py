"""PaperStudio：FastAPI 入口，微信回调与 PaperQA 异步任务。"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import BackgroundTasks, FastAPI, Request, Response

from app.config import AppConfig
from app.db import create_tables, init_db, try_mark_msg_processed, upsert_user
from app.db import get_session_factory
from app.handlers import (
    HELP_TEXT,
    handle_text_message,
    run_link_background,
    run_media_background,
    run_rag_background,
)
from app.wechat import build_text_reply, parse_wechat_xml, verify_signature
from app.web_router import indexing_worker, router as web_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
# 以下第三方库 INFO 日志噪音大（含敏感信息或无实际价值），统一调到 WARNING
for _noisy_logger in (
    "paperqa.agents.main",           # 打印完整 Settings（含 api_key）
    "paperqa.agents.main.agent_callers",  # 打印带 rich 标记的 Answer 行
    "paperqa.agents.tools",          # 逐步 Status 汇报
    "LiteLLM",                       # 每次 completion() 路由日志
    "LiteLLM Router",                # Routing strategy / 200 OK 日志
):
    logging.getLogger(_noisy_logger).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


@lru_cache
def get_config() -> AppConfig:
    return AppConfig()


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = get_config()
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    init_db(cfg.database_url)
    await create_tables()
    logger.info("PaperStudio 启动 data_dir=%s", cfg.data_dir.resolve())

    # 启动 Web UI 后台索引 worker（串行处理队列，防止 LLM 并发过载）
    worker_task = asyncio.create_task(indexing_worker(cfg))
    logger.info("Web UI 索引 worker 已创建")

    yield

    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass


app = FastAPI(title="PaperStudio", lifespan=lifespan)

# 挂载 Web UI 路由（/ui/*）
app.include_router(web_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.api_route("/", methods=["GET", "POST"])
async def wechat_root(request: Request, background_tasks: BackgroundTasks) -> Response:
    """根路径微信回调（与 /wechat 等价，适配后台 URL 未加路径的情况）。"""
    return await wechat_callback(request, background_tasks)


def _check_sig(cfg: AppConfig, request: Request) -> bool:
    sig = request.query_params.get("signature") or ""
    ts = request.query_params.get("timestamp") or ""
    nonce = request.query_params.get("nonce") or ""
    if not verify_signature(cfg.wechat_token, sig, ts, nonce):
        logger.warning("微信签名校验失败")
        return False
    return True


@app.api_route("/wechat", methods=["GET", "POST"])
async def wechat_callback(request: Request, background_tasks: BackgroundTasks) -> Response:
    cfg = get_config()
    if not cfg.wechat_token:
        logger.error("未配置 WECHAT_TOKEN")
        return Response(content="config error", status_code=500)

    if not _check_sig(cfg, request):
        return Response(content="signature error", status_code=403)

    if request.method == "GET":
        echostr = request.query_params.get("echostr") or ""
        logger.info("微信 URL 验证通过")
        return Response(content=echostr, media_type="text/plain")

    body = await request.body()
    if not body:
        return Response(content="success", media_type="text/plain")

    data = parse_wechat_xml(body)
    msg_type = data.get("MsgType", "")
    from_user = data.get("FromUserName", "")
    to_user = data.get("ToUserName", "")
    msg_id = data.get("MsgId", "")

    logger.info(
        "微信消息 MsgType=%s From=%s... MsgId=%s",
        msg_type,
        from_user[:8],
        msg_id,
    )

    factory = get_session_factory()
    async with factory() as session:
        if not await try_mark_msg_processed(session, msg_id):
            return Response(content="success", media_type="text/plain")
        await upsert_user(session, from_user)

    # 事件：关注
    if msg_type == "event":
        ev = data.get("Event", "")
        logger.info("微信事件 Event=%s", ev)
        if ev == "subscribe":
            xml = build_text_reply(
                to_user=from_user,
                from_user=to_user,
                content=HELP_TEXT + "\n\n感谢关注！可直接提问或发送「帮助」。",
            )
            return Response(content=xml.encode("utf-8"), media_type="application/xml")
        return Response(content="success", media_type="text/plain")

    # 链接消息：异步拉取
    if msg_type == "link":
        url = data.get("Url", "")
        if url and cfg.wechat_app_id and cfg.wechat_app_secret:
            background_tasks.add_task(
                run_link_background,
                cfg,
                from_user,
                url,
                app_id=cfg.wechat_app_id,
                app_secret=cfg.wechat_app_secret,
            )
            xml = build_text_reply(
                to_user=from_user,
                from_user=to_user,
                content="已收到链接，正在尝试下载并加入你的文献库…",
            )
            return Response(content=xml.encode("utf-8"), media_type="application/xml")
        xml = build_text_reply(
            to_user=from_user,
            from_user=to_user,
            content="收到链接，但未配置 WECHAT_APP_ID/SECRET，无法下载素材。",
        )
        return Response(content=xml.encode("utf-8"), media_type="application/xml")

    # 图片：下载临时素材并保存
    if msg_type == "image":
        mid = data.get("MediaId", "")
        if mid and cfg.wechat_app_id and cfg.wechat_app_secret:
            from app.wechat import download_media

            async def _img_task() -> None:
                raw, sfx = await download_media(
                    cfg.wechat_app_id, cfg.wechat_app_secret, mid
                )
                await run_media_background(
                    cfg,
                    from_user,
                    raw,
                    sfx,
                    app_id=cfg.wechat_app_id,
                    app_secret=cfg.wechat_app_secret,
                )

            background_tasks.add_task(_img_task)
            xml = build_text_reply(
                to_user=from_user,
                from_user=to_user,
                content="已收到图片，正在保存到文献目录（复杂文献请优先发 PDF 链接）。",
            )
            return Response(content=xml.encode("utf-8"), media_type="application/xml")
        xml = build_text_reply(
            to_user=from_user,
            from_user=to_user,
            content="收到图片，但未配置微信 AppId/Secret，无法拉取素材。",
        )
        return Response(content=xml.encode("utf-8"), media_type="application/xml")

    # 语音 / 视频：提示打字
    if msg_type in ("voice", "video", "shortvideo"):
        xml = build_text_reply(
            to_user=from_user,
            from_user=to_user,
            content="当前版本请直接发送文字问题，或发送论文 PDF 链接 / 使用「添加 URL」。",
        )
        return Response(content=xml.encode("utf-8"), media_type="application/xml")

    # 文本
    if msg_type == "text":
        content = data.get("Content", "")
        sync, rag_q = await handle_text_message(cfg, from_user, content)
        if rag_q:
            if not cfg.wechat_app_id or not cfg.wechat_app_secret:
                xml = build_text_reply(
                    to_user=from_user,
                    from_user=to_user,
                    content="未配置 WECHAT_APP_ID/SECRET，无法发送长回答（客服消息）。请配置后重试。",
                )
                return Response(content=xml.encode("utf-8"), media_type="application/xml")
            background_tasks.add_task(
                run_rag_background,
                cfg,
                from_user,
                rag_q,
                app_id=cfg.wechat_app_id,
                app_secret=cfg.wechat_app_secret,
            )
            ack = (
                sync
                or "思考中…"
            )
            xml = build_text_reply(
                to_user=from_user, from_user=to_user, content=ack
            )
            return Response(content=xml.encode("utf-8"), media_type="application/xml")

        xml = build_text_reply(to_user=from_user, from_user=to_user, content=sync)
        return Response(content=xml.encode("utf-8"), media_type="application/xml")

    # 其它类型
    xml = build_text_reply(
        to_user=from_user,
        from_user=to_user,
        content="暂不支持该消息类型。请发送文字、链接或图片，或输入「帮助」。",
    )
    return Response(content=xml.encode("utf-8"), media_type="application/xml")


def main() -> None:
    import uvicorn

    cfg = get_config()
    uvicorn.run(
        "app.main:app",
        host=cfg.host,
        port=cfg.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
