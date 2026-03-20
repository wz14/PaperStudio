"""微信公众平台：签名校验、XML 解析、被动回复 XML、客服消息。"""

from __future__ import annotations

import hashlib
import logging
import time
import xml.etree.ElementTree as ET
from typing import Any

import httpx

logger = logging.getLogger(__name__)

WECHAT_TOKEN_URL = "https://api.weixin.qq.com/cgi-bin/token"
WECHAT_CUSTOM_SEND_URL = "https://api.weixin.qq.com/cgi-bin/message/custom/send"
WECHAT_MEDIA_GET_URL = "https://api.weixin.qq.com/cgi-bin/media/get"


def verify_signature(token: str, signature: str, timestamp: str, nonce: str) -> bool:
    if not token or not signature:
        return False
    arr = sorted([token, timestamp, nonce])
    digest = hashlib.sha1("".join(arr).encode("utf-8")).hexdigest()
    return digest == signature


def parse_wechat_xml(body: bytes) -> dict[str, Any]:
    root = ET.fromstring(body)
    out: dict[str, Any] = {}
    for child in root:
        tag = child.tag
        text = (child.text or "").strip()
        out[tag] = text
    return out


def build_text_reply(to_user: str, from_user: str, content: str) -> str:
    """被动回复文本（UTF-8 XML）。"""
    ts = int(time.time())
    return f"""<xml>
<ToUserName><![CDATA[{to_user}]]></ToUserName>
<FromUserName><![CDATA[{from_user}]]></FromUserName>
<CreateTime>{ts}</CreateTime>
<MsgType><![CDATA[text]]></MsgType>
<Content><![CDATA[{content}]]></Content>
</xml>"""


_token_cache: dict[str, Any] = {"token": None, "expires_at": 0.0}


async def get_access_token(app_id: str, app_secret: str) -> str:
    """带简单内存缓存的 access_token。"""
    now = time.time()
    tok = _token_cache.get("token")
    exp = float(_token_cache.get("expires_at") or 0)
    if tok and now < exp - 120:
        return str(tok)

    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.get(
            WECHAT_TOKEN_URL,
            params={
                "grant_type": "client_credential",
                "appid": app_id,
                "secret": app_secret,
            },
        )
        r.raise_for_status()
        data = r.json()
    if "access_token" not in data:
        logger.error("获取 access_token 失败: %s", data)
        raise RuntimeError(f"微信 access_token 错误: {data}")
    _token_cache["token"] = data["access_token"]
    _token_cache["expires_at"] = now + int(data.get("expires_in", 7200))
    logger.info("已刷新微信 access_token")
    return str(data["access_token"])


async def send_custom_text(
    app_id: str, app_secret: str, openid: str, content: str
) -> None:
    """客服消息文本；超长拆条发送（单条上限约 2048）。"""
    token = await get_access_token(app_id, app_secret)
    chunk_size = 2000
    parts = [content[i : i + chunk_size] for i in range(0, len(content), chunk_size)]
    async with httpx.AsyncClient(timeout=60.0) as client:
        for i, part in enumerate(parts):
            payload = {
                "touser": openid,
                "msgtype": "text",
                "text": {"content": part},
            }
            r = await client.post(
                f"{WECHAT_CUSTOM_SEND_URL}?access_token={token}",
                json=payload,
            )
            data = r.json()
            if data.get("errcode", 0) != 0:
                logger.error(
                    "客服消息发送失败 part=%d/%d err=%s body=%s",
                    i + 1,
                    len(parts),
                    data,
                    part[:200],
                )
                raise RuntimeError(f"微信客服消息失败: {data}")
            logger.info(
                "客服消息已发送 part=%d/%d len=%d", i + 1, len(parts), len(part)
            )


async def download_media(
    app_id: str, app_secret: str, media_id: str
) -> tuple[bytes, str]:
    """下载临时素材，返回 (body, 建议后缀)。"""
    token = await get_access_token(app_id, app_secret)
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
        r = await client.get(
            WECHAT_MEDIA_GET_URL,
            params={"access_token": token, "media_id": media_id},
        )
        r.raise_for_status()
        body = r.content
        if body[:1] == b"{":
            logger.error("下载素材返回 JSON 错误: %s", body[:500])
            raise RuntimeError(f"微信素材下载失败: {body.decode('utf-8', errors='replace')}")
        ct = r.headers.get("content-type", "").lower()
    suffix = ".bin"
    if "jpeg" in ct or "jpg" in ct:
        suffix = ".jpg"
    elif "png" in ct:
        suffix = ".png"
    elif "pdf" in ct:
        suffix = ".pdf"
    elif "audio" in ct or "mpeg" in ct:
        suffix = ".mp3"
    logger.info("已下载临时素材 media_id=%s... suffix=%s bytes=%d", media_id[:8], suffix, len(body))
    return body, suffix
