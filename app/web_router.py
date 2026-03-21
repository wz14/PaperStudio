"""Web UI 路由：论文上传、索引管理与多用户并发支持。

功能：
- 账号密码登录（配置在 .env WEB_USERS）
- 文件上传（支持并发多用户、逐文件进度追踪）
- 索引队列（全局串行，避免 LLM API 过载）
- SSE 实时进度推送
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, AsyncGenerator

from fastapi import APIRouter, Cookie, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
import jwt as pyjwt
from jwt.exceptions import InvalidTokenError as JWTError

from app.config import AppConfig
from app.rag_service import build_settings_for_user, user_paths

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────
# 全局状态（进程内共享）
# ──────────────────────────────────────────────────────

# 索引任务队列；item = {"task_id": str, "username": str}
_index_queue: asyncio.Queue[dict] = asyncio.Queue()

# task_id -> {task_id, username, status, queued_at, started_at, finished_at, error}
_task_status: dict[str, dict] = {}

# SSE 订阅者：event -> set of asyncio.Queue
_sse_subscribers: set[asyncio.Queue] = set()


def _notify_sse(data: dict) -> None:
    """向所有 SSE 订阅者广播事件。"""
    for q in list(_sse_subscribers):
        try:
            q.put_nowait(data)
        except asyncio.QueueFull:
            pass


# ──────────────────────────────────────────────────────
# 认证工具
# ──────────────────────────────────────────────────────


def _parse_web_users(raw: str) -> dict[str, str]:
    """解析 'user1:pass1,user2:pass2' 格式为 {username: password}。"""
    users: dict[str, str] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if ":" in pair:
            u, _, p = pair.partition(":")
            users[u.strip()] = p.strip()
    return users


def _make_token(cfg: AppConfig, username: str) -> str:
    exp = datetime.now(timezone.utc) + timedelta(hours=24)
    return pyjwt.encode(
        {"sub": username, "exp": exp}, cfg.web_secret_key, algorithm="HS256"
    )


def _verify_token(cfg: AppConfig, token: str) -> str:
    payload = pyjwt.decode(token, cfg.web_secret_key, algorithms=["HS256"])
    return str(payload["sub"])


def _require_auth(cfg: AppConfig, token: str | None) -> str:
    if not token:
        raise HTTPException(status_code=401, detail="未登录")
    try:
        return _verify_token(cfg, token)
    except JWTError as exc:
        raise HTTPException(status_code=401, detail="登录已过期，请重新登录") from exc


def _web_openid(username: str) -> str:
    """Web 用户的虚拟 openid，与微信用户隔离。"""
    return f"webui_{username}"


# ──────────────────────────────────────────────────────
# 后台索引 Worker（由 main.py lifespan 启动）
# ──────────────────────────────────────────────────────


async def indexing_worker(cfg: AppConfig) -> None:
    """串行处理队列中的索引任务，防止并发 LLM 调用过载。"""
    logger.info("Web UI 索引 worker 已启动")
    while True:
        task = await _index_queue.get()
        task_id: str = task["task_id"]
        username: str = task["username"]
        logger.info("开始索引 task_id=%s username=%s", task_id, username)

        _task_status[task_id]["status"] = "indexing"
        _task_status[task_id]["started_at"] = time.time()
        _notify_sse({"type": "task_update", "task": _task_status[task_id]})

        try:
            openid = _web_openid(username)
            settings = build_settings_for_user(cfg, openid)
            # paperqa 提供的函数：构建/更新向量索引
            from paperqa.agents.search import get_directory_index  # noqa: PLC0415

            await get_directory_index(settings=settings)
            _task_status[task_id]["status"] = "done"
            _task_status[task_id]["finished_at"] = time.time()
            logger.info("索引完成 task_id=%s", task_id)
        except Exception as e:
            logger.exception("索引失败 task_id=%s username=%s", task_id, username)
            _task_status[task_id]["status"] = "error"
            _task_status[task_id]["error"] = str(e)
        finally:
            _index_queue.task_done()
            _notify_sse({"type": "task_update", "task": _task_status[task_id]})


# ──────────────────────────────────────────────────────
# FastAPI Router
# ──────────────────────────────────────────────────────

router = APIRouter(prefix="/ui")


@router.get("/", response_class=HTMLResponse)
async def serve_ui() -> HTMLResponse:
    """返回前端单页应用。"""
    return HTMLResponse(content=_HTML_PAGE)


# ── Auth ──────────────────────────────────────────────


@router.post("/api/login")
async def login(request: Request) -> Response:
    cfg = AppConfig()
    data = await request.json()
    username = str(data.get("username", "")).strip()
    password = str(data.get("password", ""))

    users = _parse_web_users(cfg.web_users)
    if not users:
        raise HTTPException(status_code=500, detail="服务器未配置 WEB_USERS，请联系管理员")
    if username not in users or users[username] != password:
        logger.warning("登录失败 username=%s", username)
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    token = _make_token(cfg, username)
    resp = JSONResponse({"ok": True, "username": username})
    resp.set_cookie("auth_token", token, httponly=True, samesite="lax", max_age=86400)
    logger.info("用户登录 username=%s", username)
    return resp


@router.post("/api/logout")
async def logout() -> Response:
    resp = JSONResponse({"ok": True})
    resp.delete_cookie("auth_token")
    return resp


@router.get("/api/me")
async def me(auth_token: Annotated[str | None, Cookie()] = None) -> JSONResponse:
    cfg = AppConfig()
    username = _require_auth(cfg, auth_token)
    return JSONResponse({"username": username})


# ── 文件上传 ───────────────────────────────────────────


@router.post("/api/upload")
async def upload_files(
    files: list[UploadFile] = File(...),
    auth_token: Annotated[str | None, Cookie()] = None,
) -> JSONResponse:
    cfg = AppConfig()
    username = _require_auth(cfg, auth_token)
    openid = _web_openid(username)
    papers_dir, _ = user_paths(cfg, openid)
    papers_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for f in files:
        data = await f.read()
        original_name = f.filename or "unknown"
        content_hash = hashlib.sha256(data).hexdigest()[:8]
        # 保留可读文件名（最多50字符），追加哈希避免重复
        stem = Path(original_name).stem[:50]
        suffix = Path(original_name).suffix.lower() or ".pdf"
        dest_name = f"{stem}_{content_hash}{suffix}"
        dest = papers_dir / dest_name
        dest.write_bytes(data)
        saved.append({"name": dest_name, "original": original_name, "size": len(data)})
        logger.info(
            "上传文件 username=%s name=%s bytes=%d", username, dest_name, len(data)
        )

    return JSONResponse({"saved": saved})


# ── 索引队列 ───────────────────────────────────────────


@router.post("/api/index")
async def trigger_index(
    auth_token: Annotated[str | None, Cookie()] = None,
) -> JSONResponse:
    """将当前用户的文献库加入索引队列。"""
    cfg = AppConfig()
    username = _require_auth(cfg, auth_token)

    task_id = uuid.uuid4().hex[:8]
    _task_status[task_id] = {
        "task_id": task_id,
        "username": username,
        "status": "queued",
        "queued_at": time.time(),
        "started_at": None,
        "finished_at": None,
        "error": None,
    }
    await _index_queue.put({"task_id": task_id, "username": username})
    queue_pos = _index_queue.qsize()

    logger.info(
        "添加索引任务 task_id=%s username=%s queue_pos=%d", task_id, username, queue_pos
    )
    _notify_sse({"type": "task_update", "task": _task_status[task_id]})
    return JSONResponse({"task_id": task_id, "queue_position": queue_pos})


@router.get("/api/queue")
async def get_queue_status(
    auth_token: Annotated[str | None, Cookie()] = None,
) -> JSONResponse:
    cfg = AppConfig()
    _require_auth(cfg, auth_token)

    # 返回最近 30 条任务，按时间降序
    tasks = sorted(
        _task_status.values(), key=lambda t: t.get("queued_at", 0), reverse=True
    )[:30]
    return JSONResponse({"tasks": tasks, "queue_size": _index_queue.qsize()})


# ── 论文管理 ───────────────────────────────────────────


@router.get("/api/papers")
async def list_papers(
    auth_token: Annotated[str | None, Cookie()] = None,
) -> JSONResponse:
    cfg = AppConfig()
    username = _require_auth(cfg, auth_token)
    openid = _web_openid(username)
    papers_dir, _ = user_paths(cfg, openid)

    papers = []
    if papers_dir.exists():
        for f in sorted(papers_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if f.is_file() and not f.name.startswith("."):
                papers.append(
                    {
                        "name": f.name,
                        "size": f.stat().st_size,
                        "mtime": f.stat().st_mtime,
                    }
                )
    return JSONResponse({"papers": papers})


@router.delete("/api/papers/{filename}")
async def delete_paper(
    filename: str,
    auth_token: Annotated[str | None, Cookie()] = None,
) -> JSONResponse:
    cfg = AppConfig()
    username = _require_auth(cfg, auth_token)
    openid = _web_openid(username)
    papers_dir, _ = user_paths(cfg, openid)

    # 路径穿越防护
    target = (papers_dir / filename).resolve()
    if not str(target).startswith(str(papers_dir.resolve())):
        raise HTTPException(status_code=400, detail="非法路径")
    if not target.exists():
        raise HTTPException(status_code=404, detail="文件不存在")

    target.unlink()
    logger.info("删除论文 username=%s name=%s", username, filename)
    return JSONResponse({"ok": True})


# ── SSE 实时推送 ───────────────────────────────────────


@router.get("/api/events")
async def sse_events(
    auth_token: Annotated[str | None, Cookie()] = None,
) -> StreamingResponse:
    """Server-Sent Events，推送索引任务状态变更。"""
    cfg = AppConfig()
    _require_auth(cfg, auth_token)

    async def event_stream() -> AsyncGenerator[str, None]:
        q: asyncio.Queue = asyncio.Queue(maxsize=50)
        _sse_subscribers.add(q)
        try:
            # 首次连接推送当前完整状态
            tasks = sorted(
                _task_status.values(), key=lambda t: t.get("queued_at", 0), reverse=True
            )[:30]
            yield f"data: {__import__('json').dumps({'type': 'init', 'tasks': tasks})}\n\n"

            while True:
                try:
                    data = await asyncio.wait_for(q.get(), timeout=30.0)
                    yield f"data: {__import__('json').dumps(data)}\n\n"
                except asyncio.TimeoutError:
                    # 心跳防断连
                    yield ": heartbeat\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            _sse_subscribers.discard(q)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ──────────────────────────────────────────────────────
# 内嵌 HTML 前端
# ──────────────────────────────────────────────────────

_HTML_PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>PaperStudio</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    * { font-family: 'Inter', system-ui, -apple-system, sans-serif; }

    /* 登录页渐变背景 */
    .login-bg {
      background: linear-gradient(135deg, #1e1b4b 0%, #312e81 40%, #4c1d95 100%);
    }

    /* 上传拖拽区 */
    .dropzone {
      border: 2px dashed #c7d2fe;
      transition: all 0.25s ease;
    }
    .dropzone.drag-over {
      border-color: #6366f1;
      background: #eef2ff;
      transform: scale(1.01);
    }

    /* 进度条动画 */
    .progress-bar {
      transition: width 0.3s ease;
    }

    /* 索引状态徽章 */
    .badge-queued   { background: #fef3c7; color: #92400e; }
    .badge-indexing { background: #dbeafe; color: #1e40af; }
    .badge-done     { background: #d1fae5; color: #065f46; }
    .badge-error    { background: #fee2e2; color: #991b1b; }

    /* 旋转动画 */
    @keyframes spin { to { transform: rotate(360deg); } }
    .spin { animation: spin 1s linear infinite; }

    /* 侧边栏滚动 */
    .sidebar-scroll { overflow-y: auto; max-height: calc(100vh - 140px); }

    /* 卡片悬停 */
    .paper-item:hover { background: #f1f5f9; }

    /* 自定义滚动条 */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #f1f5f9; }
    ::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }

    /* 淡入动画 */
    @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
    .fade-in { animation: fadeIn 0.3s ease forwards; }

    /* 脉冲动画（索引中状态） */
    @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: .5; } }
    .pulse { animation: pulse 1.5s ease-in-out infinite; }
  </style>
</head>
<body class="bg-slate-50 min-h-screen">

<!-- ═══════════════════════════════════════════════════ 登录页 ══ -->
<div id="login-page" class="hidden login-bg min-h-screen flex items-center justify-center p-4">
  <div class="bg-white rounded-2xl shadow-2xl p-8 w-full max-w-md fade-in">
    <div class="text-center mb-8">
      <div class="inline-flex items-center justify-center w-16 h-16 bg-indigo-100 rounded-2xl mb-4">
        <svg class="w-9 h-9 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
            d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.746 0 3.332.477 4.5 1.253v13C19.832 18.477 18.246 18 16.5 18c-1.746 0-3.332.477-4.5 1.253"/>
        </svg>
      </div>
      <h1 class="text-2xl font-bold text-slate-800">PaperStudio</h1>
      <p class="text-slate-500 text-sm mt-1">论文知识库管理平台</p>
    </div>

    <form id="login-form" class="space-y-4" onsubmit="return false">
      <div>
        <label class="block text-sm font-medium text-slate-700 mb-1">用户名</label>
        <input id="login-username" type="text" autocomplete="username"
          class="w-full px-4 py-2.5 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition"
          placeholder="请输入用户名" />
      </div>
      <div>
        <label class="block text-sm font-medium text-slate-700 mb-1">密码</label>
        <input id="login-password" type="password" autocomplete="current-password"
          class="w-full px-4 py-2.5 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition"
          placeholder="请输入密码" />
      </div>
      <div id="login-error" class="hidden text-sm text-red-600 bg-red-50 border border-red-200 rounded-lg px-3 py-2"></div>
      <button onclick="doLogin()"
        class="w-full py-2.5 px-4 bg-indigo-600 hover:bg-indigo-700 text-white font-semibold rounded-lg transition shadow-sm active:scale-95">
        登录
      </button>
    </form>
  </div>
</div>

<!-- ═══════════════════════════════════════════════════ 主界面 ══ -->
<div id="dashboard" class="hidden min-h-screen flex flex-col">

  <!-- 顶部导航 -->
  <nav class="bg-white border-b border-slate-200 px-6 py-3 flex items-center justify-between shadow-sm sticky top-0 z-10">
    <div class="flex items-center gap-3">
      <div class="bg-indigo-600 text-white rounded-lg p-1.5">
        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
            d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.746 0 3.332.477 4.5 1.253v13C19.832 18.477 18.246 18 16.5 18c-1.746 0-3.332.477-4.5 1.253"/>
        </svg>
      </div>
      <span class="font-bold text-slate-800 text-lg">PaperStudio</span>
    </div>
    <div class="flex items-center gap-4">
      <div class="flex items-center gap-2 text-sm text-slate-600">
        <div class="w-7 h-7 bg-indigo-100 text-indigo-700 rounded-full flex items-center justify-center font-semibold text-xs" id="user-avatar">A</div>
        <span id="nav-username" class="font-medium"></span>
      </div>
      <button onclick="doLogout()"
        class="text-sm text-slate-500 hover:text-red-600 flex items-center gap-1 px-3 py-1.5 rounded-lg hover:bg-red-50 transition">
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
            d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1"/>
        </svg>
        退出
      </button>
    </div>
  </nav>

  <!-- 主体布局 -->
  <div class="flex flex-1 overflow-hidden">

    <!-- 左侧边栏：已索引论文 -->
    <aside class="w-72 bg-white border-r border-slate-200 flex flex-col">
      <div class="px-4 py-3 border-b border-slate-100 flex items-center justify-between">
        <h2 class="text-sm font-semibold text-slate-700 flex items-center gap-2">
          <svg class="w-4 h-4 text-indigo-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
          </svg>
          文献库
        </h2>
        <span id="paper-count" class="text-xs bg-indigo-50 text-indigo-600 rounded-full px-2 py-0.5 font-medium">0 篇</span>
      </div>
      <div class="sidebar-scroll px-2 py-2 flex-1">
        <div id="papers-list" class="space-y-1">
          <div class="text-center py-8 text-slate-400 text-sm" id="papers-empty">
            <svg class="w-10 h-10 mx-auto mb-2 text-slate-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M9 13h6m-3-3v6m5 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
            </svg>
            暂无论文
          </div>
        </div>
      </div>
    </aside>

    <!-- 右侧主区域 -->
    <main class="flex-1 overflow-y-auto p-6 space-y-6">

      <!-- 上传区域 -->
      <div class="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
        <h2 class="text-base font-semibold text-slate-800 mb-4 flex items-center gap-2">
          <svg class="w-5 h-5 text-indigo-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"/>
          </svg>
          上传论文
        </h2>

        <!-- 拖拽区 -->
        <div id="dropzone"
          class="dropzone rounded-xl bg-slate-50 p-10 text-center cursor-pointer mb-4"
          onclick="document.getElementById('file-input').click()"
          ondragover="handleDragOver(event)"
          ondragleave="handleDragLeave(event)"
          ondrop="handleDrop(event)">
          <input type="file" id="file-input" multiple accept=".pdf,.html,.txt" class="hidden" onchange="handleFileSelect(event)" />
          <div class="flex flex-col items-center gap-3 pointer-events-none">
            <div class="w-14 h-14 bg-indigo-100 rounded-2xl flex items-center justify-center">
              <svg class="w-7 h-7 text-indigo-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"/>
              </svg>
            </div>
            <div>
              <p class="text-slate-700 font-medium">拖拽文件到这里，或 <span class="text-indigo-600">点击选择</span></p>
              <p class="text-slate-400 text-sm mt-1">支持 PDF、HTML、TXT 格式，可多选</p>
            </div>
          </div>
        </div>

        <!-- 已选文件列表 -->
        <div id="file-queue" class="space-y-2 mb-4 hidden"></div>

        <!-- 操作按钮 -->
        <div class="flex items-center gap-3 flex-wrap">
          <button id="upload-btn" onclick="uploadAllFiles()" disabled
            class="flex items-center gap-2 px-5 py-2.5 bg-indigo-600 hover:bg-indigo-700 disabled:bg-slate-300 disabled:cursor-not-allowed text-white font-medium rounded-lg transition text-sm shadow-sm active:scale-95">
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"/>
            </svg>
            上传文件
          </button>
          <button id="index-btn" onclick="triggerIndex()" disabled
            class="flex items-center gap-2 px-5 py-2.5 bg-emerald-600 hover:bg-emerald-700 disabled:bg-slate-300 disabled:cursor-not-allowed text-white font-medium rounded-lg transition text-sm shadow-sm active:scale-95">
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z"/>
            </svg>
            完成并建立索引
          </button>
          <button id="clear-btn" onclick="clearFileQueue()" class="hidden text-sm text-slate-500 hover:text-slate-700 px-3 py-2.5 rounded-lg hover:bg-slate-100 transition">
            清空选择
          </button>
        </div>

        <!-- 上传整体提示 -->
        <div id="upload-hint" class="hidden mt-3 text-sm text-slate-500"></div>
      </div>

      <!-- 索引队列 -->
      <div class="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
        <div class="flex items-center justify-between mb-4">
          <h2 class="text-base font-semibold text-slate-800 flex items-center gap-2">
            <svg class="w-5 h-5 text-amber-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"/>
            </svg>
            索引队列
          </h2>
          <div class="flex items-center gap-2">
            <span id="queue-size-badge" class="text-xs bg-amber-50 text-amber-700 rounded-full px-2 py-0.5 font-medium hidden">等待中</span>
            <div id="queue-spinner" class="hidden w-4 h-4 border-2 border-indigo-500 border-t-transparent rounded-full spin"></div>
          </div>
        </div>
        <div id="queue-list" class="space-y-2">
          <div class="text-center py-6 text-slate-400 text-sm" id="queue-empty">
            <svg class="w-8 h-8 mx-auto mb-2 text-slate-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"/>
            </svg>
            暂无索引任务
          </div>
        </div>
      </div>

    </main>
  </div>
</div>

<script>
// ═══════════════════════════════ 状态 ════════════════════════════
let currentUser = null;
let pendingFiles = [];      // {file: File, status: 'pending'|'uploading'|'done'|'error', progress: 0-100, savedName: string}
let uploadDone = false;     // 本批次是否全部上传完成
let eventSource = null;
let papersList = [];

// ═══════════════════════════════ 初始化 ══════════════════════════
window.addEventListener('DOMContentLoaded', async () => {
  try {
    const r = await fetch('/ui/api/me', { credentials: 'include' });
    if (r.ok) {
      const d = await r.json();
      showDashboard(d.username);
    } else {
      showLogin();
    }
  } catch {
    showLogin();
  }
});

// ═══════════════════════════════ 视图切换 ════════════════════════
function showLogin() {
  document.getElementById('login-page').classList.remove('hidden');
  document.getElementById('dashboard').classList.add('hidden');
  document.getElementById('login-username').focus();
}

function showDashboard(username) {
  currentUser = username;
  document.getElementById('login-page').classList.add('hidden');
  document.getElementById('dashboard').classList.remove('hidden');
  document.getElementById('nav-username').textContent = username;
  document.getElementById('user-avatar').textContent = username.charAt(0).toUpperCase();
  loadPapers();
  startSSE();
}

// ═══════════════════════════════ 登录/退出 ═══════════════════════
async function doLogin() {
  const username = document.getElementById('login-username').value.trim();
  const password = document.getElementById('login-password').value;
  const errEl = document.getElementById('login-error');
  errEl.classList.add('hidden');

  if (!username || !password) {
    errEl.textContent = '请填写用户名和密码';
    errEl.classList.remove('hidden');
    return;
  }

  try {
    const r = await fetch('/ui/api/login', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({username, password}),
      credentials: 'include',
    });
    if (r.ok) {
      const d = await r.json();
      document.getElementById('login-password').value = '';
      showDashboard(d.username);
    } else {
      const d = await r.json().catch(() => ({detail: '登录失败'}));
      errEl.textContent = d.detail || '用户名或密码错误';
      errEl.classList.remove('hidden');
    }
  } catch {
    errEl.textContent = '网络错误，请稍后重试';
    errEl.classList.remove('hidden');
  }
}

async function doLogout() {
  if (eventSource) { eventSource.close(); eventSource = null; }
  await fetch('/ui/api/logout', { method: 'POST', credentials: 'include' });
  currentUser = null;
  pendingFiles = [];
  uploadDone = false;
  showLogin();
}

// 回车登录
document.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !document.getElementById('login-page').classList.contains('hidden')) {
    doLogin();
  }
});

// ═══════════════════════════════ 文件选择 ════════════════════════
function handleFileSelect(e) {
  addFiles(Array.from(e.target.files));
  e.target.value = '';
}

function handleDragOver(e) {
  e.preventDefault();
  document.getElementById('dropzone').classList.add('drag-over');
}

function handleDragLeave(e) {
  document.getElementById('dropzone').classList.remove('drag-over');
}

function handleDrop(e) {
  e.preventDefault();
  document.getElementById('dropzone').classList.remove('drag-over');
  const files = Array.from(e.dataTransfer.files).filter(f =>
    f.name.match(/\\.(pdf|html?|txt)$/i)
  );
  if (files.length) {
    addFiles(files);
  } else {
    showHint('⚠️ 请上传 PDF、HTML 或 TXT 格式的文件');
  }
}

function addFiles(files) {
  const validExts = /\\.(pdf|html?|txt)$/i;
  for (const f of files) {
    if (!validExts.test(f.name)) {
      showHint(`⚠️ 跳过不支持的文件: ${f.name}`);
      continue;
    }
    // 避免重复添加
    if (pendingFiles.some(p => p.file.name === f.name && p.file.size === f.size)) continue;
    pendingFiles.push({ file: f, status: 'pending', progress: 0, savedName: '' });
  }
  uploadDone = false;
  renderFileQueue();
  updateButtons();
}

function clearFileQueue() {
  pendingFiles = [];
  uploadDone = false;
  renderFileQueue();
  updateButtons();
  document.getElementById('upload-hint').classList.add('hidden');
}

// ═══════════════════════════════ 渲染文件列表 ════════════════════
function renderFileQueue() {
  const container = document.getElementById('file-queue');
  if (pendingFiles.length === 0) {
    container.classList.add('hidden');
    container.innerHTML = '';
    return;
  }
  container.classList.remove('hidden');
  container.innerHTML = pendingFiles.map((item, i) => {
    const size = formatSize(item.file.size);
    const statusIcon = {
      pending: '<span class="text-slate-400">⏸</span>',
      uploading: '<span class="text-indigo-500 spin inline-block">⟳</span>',
      done: '<span class="text-emerald-500">✓</span>',
      error: '<span class="text-red-500">✗</span>',
    }[item.status] || '';

    const barColor = item.status === 'done' ? 'bg-emerald-500'
      : item.status === 'error' ? 'bg-red-400' : 'bg-indigo-500';

    return `
      <div class="flex items-center gap-3 px-3 py-2 bg-slate-50 rounded-lg border border-slate-100 text-sm">
        <svg class="w-4 h-4 text-slate-400 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
        </svg>
        <div class="flex-1 min-w-0">
          <div class="flex items-center justify-between mb-1">
            <span class="truncate text-slate-700 font-medium" title="${item.file.name}">${item.file.name}</span>
            <span class="text-slate-400 text-xs shrink-0 ml-2">${size}</span>
          </div>
          <div class="w-full bg-slate-200 rounded-full h-1.5">
            <div class="progress-bar ${barColor} h-1.5 rounded-full" style="width:${item.progress}%"></div>
          </div>
        </div>
        <span class="text-base shrink-0">${statusIcon}</span>
        ${item.status === 'pending' ? `<button onclick="removeFile(${i})" class="text-slate-300 hover:text-red-400 transition shrink-0">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/>
          </svg>
        </button>` : ''}
      </div>
    `;
  }).join('');
}

function removeFile(i) {
  pendingFiles.splice(i, 1);
  renderFileQueue();
  updateButtons();
}

function updateButtons() {
  const hasPending = pendingFiles.some(f => f.status === 'pending');
  const allDone = pendingFiles.length > 0 && pendingFiles.every(f => f.status === 'done');

  document.getElementById('upload-btn').disabled = !hasPending;
  document.getElementById('index-btn').disabled = !(allDone || uploadDone);
  document.getElementById('clear-btn').classList.toggle('hidden', pendingFiles.length === 0);
}

// ═══════════════════════════════ 文件上传 ════════════════════════
async function uploadAllFiles() {
  const toUpload = pendingFiles.filter(f => f.status === 'pending');
  if (toUpload.length === 0) return;

  document.getElementById('upload-btn').disabled = true;
  showHint(`正在上传 ${toUpload.length} 个文件...`);

  // 逐个上传以实现独立进度条
  for (const item of toUpload) {
    await uploadSingleFile(item);
  }

  const allDone = pendingFiles.every(f => f.status === 'done');
  const hasError = pendingFiles.some(f => f.status === 'error');

  if (allDone) {
    uploadDone = true;
    showHint('✅ 全部上传完成！点击「完成并建立索引」将文献加入知识库。');
  } else if (hasError) {
    showHint('⚠️ 部分文件上传失败，请检查后重试。');
  }

  updateButtons();
  loadPapers();
}

function uploadSingleFile(item) {
  return new Promise(resolve => {
    item.status = 'uploading';
    item.progress = 0;
    renderFileQueue();

    const formData = new FormData();
    formData.append('files', item.file);

    const xhr = new XMLHttpRequest();
    xhr.upload.addEventListener('progress', e => {
      if (e.lengthComputable) {
        item.progress = Math.round((e.loaded / e.total) * 95);
        renderFileQueue();
      }
    });

    xhr.addEventListener('load', () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        item.status = 'done';
        item.progress = 100;
      } else {
        item.status = 'error';
        item.progress = 0;
      }
      renderFileQueue();
      resolve();
    });

    xhr.addEventListener('error', () => {
      item.status = 'error';
      renderFileQueue();
      resolve();
    });

    xhr.open('POST', '/ui/api/upload');
    xhr.withCredentials = true;
    xhr.send(formData);
  });
}

// ═══════════════════════════════ 触发索引 ════════════════════════
async function triggerIndex() {
  document.getElementById('index-btn').disabled = true;
  try {
    const r = await fetch('/ui/api/index', {
      method: 'POST',
      credentials: 'include',
    });
    if (r.ok) {
      const d = await r.json();
      const pos = d.queue_position;
      showHint(pos > 1
        ? `📬 已加入索引队列（前方还有 ${pos - 1} 个任务）`
        : '🔄 索引任务已提交，正在处理...'
      );
      // 上传完成后清空文件列表，准备下一批
      setTimeout(() => {
        pendingFiles = [];
        uploadDone = false;
        renderFileQueue();
        updateButtons();
      }, 1500);
    } else {
      const d = await r.json().catch(() => ({detail: '提交失败'}));
      showHint('❌ 提交失败: ' + (d.detail || '未知错误'));
      document.getElementById('index-btn').disabled = false;
    }
  } catch {
    showHint('❌ 网络错误');
    document.getElementById('index-btn').disabled = false;
  }
}

function showHint(msg) {
  const el = document.getElementById('upload-hint');
  el.textContent = msg;
  el.classList.remove('hidden');
}

// ═══════════════════════════════ 论文列表 ════════════════════════
async function loadPapers() {
  try {
    const r = await fetch('/ui/api/papers', { credentials: 'include' });
    if (!r.ok) return;
    const d = await r.json();
    papersList = d.papers || [];
    renderPapers();
  } catch {
    // 忽略
  }
}

function renderPapers() {
  const container = document.getElementById('papers-list');
  const empty = document.getElementById('papers-empty');
  document.getElementById('paper-count').textContent = papersList.length + ' 篇';

  if (papersList.length === 0) {
    empty.classList.remove('hidden');
    // 清除论文条目但保留 empty 提示
    Array.from(container.children).forEach(c => {
      if (c !== empty) c.remove();
    });
    return;
  }

  empty.classList.add('hidden');
  container.innerHTML = papersList.map(p => `
    <div class="paper-item flex items-start gap-2 px-2 py-1.5 rounded-lg cursor-default group">
      <svg class="w-4 h-4 text-red-400 mt-0.5 shrink-0" fill="currentColor" viewBox="0 0 24 24">
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8l-6-6zm-1 1.5L18.5 9H13V3.5zM6 20V4h5v7h7v9H6z"/>
      </svg>
      <div class="flex-1 min-w-0">
        <p class="text-xs text-slate-700 truncate font-medium" title="${p.name}">${p.name}</p>
        <p class="text-xs text-slate-400">${formatSize(p.size)}</p>
      </div>
      <button onclick="deletePaper('${p.name}')"
        class="opacity-0 group-hover:opacity-100 text-slate-300 hover:text-red-500 transition shrink-0 mt-0.5">
        <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"/>
        </svg>
      </button>
    </div>
  `).join('');
}

async function deletePaper(name) {
  if (!confirm(`确认删除 "${name}" 吗？此操作不可撤销。`)) return;
  try {
    const r = await fetch(`/ui/api/papers/${encodeURIComponent(name)}`, {
      method: 'DELETE',
      credentials: 'include',
    });
    if (r.ok) loadPapers();
    else alert('删除失败');
  } catch {
    alert('网络错误');
  }
}

// ═══════════════════════════════ SSE & 队列 ══════════════════════
function startSSE() {
  if (eventSource) eventSource.close();
  eventSource = new EventSource('/ui/api/events', { withCredentials: true });

  eventSource.onmessage = e => {
    try {
      const data = JSON.parse(e.data);
      if (data.type === 'init') {
        renderQueue(data.tasks);
      } else if (data.type === 'task_update') {
        updateTaskInQueue(data.task);
        // 索引完成后刷新论文列表
        if (data.task.status === 'done' || data.task.status === 'error') {
          loadPapers();
        }
      }
    } catch {}
  };

  eventSource.onerror = () => {
    // SSE 断开后轮询兜底
    setTimeout(pollQueue, 3000);
  };
}

async function pollQueue() {
  try {
    const r = await fetch('/ui/api/queue', { credentials: 'include' });
    if (r.ok) {
      const d = await r.json();
      renderQueue(d.tasks);
      updateQueueSizeBadge(d.queue_size);
    }
  } catch {}
}

let queueTasks = [];

function renderQueue(tasks) {
  queueTasks = tasks;
  const container = document.getElementById('queue-list');
  const empty = document.getElementById('queue-empty');

  // 活跃任务（非 done+error 的）排前面
  const active = tasks.filter(t => t.status !== 'done' && t.status !== 'error');
  const recent = tasks.filter(t => t.status === 'done' || t.status === 'error').slice(0, 5);
  const display = [...active, ...recent];

  const hasActive = active.length > 0;
  document.getElementById('queue-spinner').classList.toggle('hidden', !hasActive);
  updateQueueSizeBadge(active.length);

  if (display.length === 0) {
    empty.classList.remove('hidden');
    Array.from(container.children).forEach(c => { if (c !== empty) c.remove(); });
    return;
  }

  empty.classList.add('hidden');
  container.innerHTML = display.map(task => {
    const label = {
      queued:   '<span class="badge-queued badge text-xs font-medium px-2 py-0.5 rounded-full">⏳ 排队中</span>',
      indexing: '<span class="badge-indexing badge text-xs font-medium px-2 py-0.5 rounded-full pulse">⟳ 索引中</span>',
      done:     '<span class="badge-done badge text-xs font-medium px-2 py-0.5 rounded-full">✓ 完成</span>',
      error:    '<span class="badge-error badge text-xs font-medium px-2 py-0.5 rounded-full">✗ 失败</span>',
    }[task.status] || '';

    const duration = task.finished_at && task.started_at
      ? `耗时 ${(task.finished_at - task.started_at).toFixed(0)}s`
      : task.started_at ? `进行中 ${(Date.now()/1000 - task.started_at).toFixed(0)}s`
      : '';

    const errorMsg = task.error ? `<p class="text-xs text-red-500 mt-1 truncate">${task.error}</p>` : '';

    return `
      <div class="flex items-center gap-3 px-3 py-2.5 bg-slate-50 rounded-lg border border-slate-100 text-sm">
        <div class="flex-1 min-w-0">
          <div class="flex items-center gap-2">
            <span class="font-medium text-slate-700">${task.username}</span>
            ${label}
          </div>
          <div class="flex items-center gap-2 mt-0.5">
            <span class="text-xs text-slate-400">任务 ${task.task_id}</span>
            ${duration ? `<span class="text-xs text-slate-400">· ${duration}</span>` : ''}
          </div>
          ${errorMsg}
        </div>
      </div>
    `;
  }).join('');
}

function updateTaskInQueue(task) {
  const idx = queueTasks.findIndex(t => t.task_id === task.task_id);
  if (idx >= 0) queueTasks[idx] = task;
  else queueTasks.unshift(task);
  renderQueue(queueTasks);
}

function updateQueueSizeBadge(size) {
  const badge = document.getElementById('queue-size-badge');
  if (size > 0) {
    badge.textContent = `${size} 个等待中`;
    badge.classList.remove('hidden');
  } else {
    badge.classList.add('hidden');
  }
}

// ═══════════════════════════════ 工具 ════════════════════════════
function formatSize(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / 1048576).toFixed(1) + ' MB';
}
</script>
</body>
</html>"""
