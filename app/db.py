"""SQLite：用户记录、对话消息、微信消息幂等。"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from sqlalchemy import DateTime, Integer, String, Text, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class WxUser(Base):
    __tablename__ = "wx_users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    openid: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class ChatMessage(Base):
    """多用户对话历史（便于后续扩展多轮与审计）。"""

    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    openid: Mapped[str] = mapped_column(String(64), index=True)
    role: Mapped[str] = mapped_column(String(16))  # user | assistant | system
    content: Mapped[str] = mapped_column(Text())
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class ProcessedWechatMessage(Base):
    """微信可能重试同一 MsgId，用于幂等。"""

    __tablename__ = "processed_wechat_messages"

    msg_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    processed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


_engine = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    if _session_factory is None:
        raise RuntimeError("init_db 未调用")
    return _session_factory


def init_db(database_url: str) -> None:
    global _engine, _session_factory
    _engine = create_async_engine(database_url, echo=False)
    _session_factory = async_sessionmaker(_engine, expire_on_commit=False)


async def create_tables() -> None:
    if _engine is None:
        raise RuntimeError("init_db 未调用")
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("数据库表已就绪")


async def upsert_user(session: AsyncSession, openid: str) -> None:
    r = await session.execute(select(WxUser).where(WxUser.openid == openid))
    if r.scalar_one_or_none() is None:
        session.add(WxUser(openid=openid))
        await session.commit()
        logger.info("新微信用户: openid=%s...", openid[:8])


async def append_message(
    session: AsyncSession, openid: str, role: str, content: str
) -> None:
    session.add(ChatMessage(openid=openid, role=role, content=content))
    await session.commit()


async def recent_messages(
    session: AsyncSession, openid: str, limit: int = 10
) -> list[tuple[str, str]]:
    r = await session.execute(
        select(ChatMessage.role, ChatMessage.content)
        .where(ChatMessage.openid == openid)
        .order_by(ChatMessage.id.desc())
        .limit(limit)
    )
    rows = list(r.all())
    rows.reverse()
    return [(role, content) for role, content in rows]


async def try_mark_msg_processed(session: AsyncSession, msg_id: str) -> bool:
    """若 msg_id 已存在则返回 False（应跳过业务）；否则插入并返回 True。msg_id 空则不做幂等。"""
    if not msg_id:
        return True
    session.add(ProcessedWechatMessage(msg_id=msg_id))
    try:
        await session.commit()
        return True
    except IntegrityError:
        await session.rollback()
        logger.warning("重复微信消息 MsgId=%s，跳过", msg_id)
        return False
