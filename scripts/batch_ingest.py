"""批量导入论文到 RAG 数据库。

支持三种输入方式：
  1. 命令行直接传入 URL 或本地文件路径
  2. 通过 --list 指定一个文本文件，每行一条 URL 或文件路径
  3. 通过 --dir 指定一个目录，递归扫描其中所有 PDF/HTML/TXT 文件

用法示例：
  # 批量导入 URL
  python scripts/batch_ingest.py --openid oXXXX \
      https://arxiv.org/pdf/2310.01234 \
      https://arxiv.org/pdf/2310.05678

  # 从文件列表导入（每行一条 URL 或路径，# 开头为注释）
  python scripts/batch_ingest.py --openid oXXXX --list papers.txt

  # 扫描本地目录批量导入
  python scripts/batch_ingest.py --openid oXXXX --dir ./my_papers/

  # 组合使用
  python scripts/batch_ingest.py --openid oXXXX --list papers.txt --dir ./extra/
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# 将项目根目录加入 sys.path，使 `app` 包可直接导入
_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_ROOT))

from app.config import AppConfig  # pylint: disable=import-error,wrong-import-position
from app.rag_service import (  # pylint: disable=import-error,wrong-import-position
    ingest_url,
    save_uploaded_bytes,
    user_dir_hash,
    user_paths,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("batch_ingest")

# 支持的本地文件扩展名
_SUPPORTED_SUFFIXES = {".pdf", ".html", ".htm", ".txt"}


def _collect_from_dir(directory: Path) -> list[Path]:
    """递归扫描目录，收集所有支持的文献文件。"""
    if not directory.exists():
        raise FileNotFoundError(f"目录不存在: {directory}")
    files = [
        f
        for f in directory.rglob("*")
        if f.is_file() and f.suffix.lower() in _SUPPORTED_SUFFIXES
    ]
    logger.info("扫描目录 %s，找到 %d 个文件", directory, len(files))
    return sorted(files)


def _collect_from_list_file(list_file: Path) -> list[str]:
    """读取文本文件，返回去除注释和空行后的每行内容（URL 或路径）。"""
    if not list_file.exists():
        raise FileNotFoundError(f"列表文件不存在: {list_file}")
    lines: list[str] = []
    for raw in list_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            lines.append(line)
    logger.info("从 %s 读取 %d 条记录", list_file, len(lines))
    return lines


async def _ingest_one(cfg: AppConfig, openid: str, source: str) -> tuple[str, bool, str]:
    """
    导入单条来源（URL 或本地路径）。

    Returns:
        (source, success, message) 三元组。
    """
    # 判断是否是 URL
    if source.startswith("http://") or source.startswith("https://"):
        try:
            msg = await ingest_url(cfg, openid, source)
            return source, True, msg
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("下载失败 url=%s err=%s", source, e)
            return source, False, str(e)

    # 本地文件
    path = Path(source)
    if not path.is_absolute():
        # 相对路径基于脚本执行时的工作目录解析
        path = Path.cwd() / path

    if not path.exists():
        return source, False, f"文件不存在: {path}"
    if path.suffix.lower() not in _SUPPORTED_SUFFIXES:
        return source, False, f"不支持的文件类型: {path.suffix}"

    try:
        data = path.read_bytes()
        msg = await save_uploaded_bytes(cfg, openid, data, path.suffix.lower())
        return source, True, msg
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("读取文件失败 path=%s err=%s", path, e)
        return source, False, str(e)


async def batch_ingest(
    cfg: AppConfig,
    openid: str,
    sources: list[str],
    concurrency: int = 3,
) -> None:
    """
    并发批量导入，concurrency 控制同时进行的下载/写入任务数量。
    过高的并发可能触发外部网站限速，默认值 3 较为保守。
    """
    papers_dir, _ = user_paths(cfg, openid)
    papers_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "开始批量导入: openid_hash=%s 共 %d 条，并发数=%d，papers_dir=%s",
        user_dir_hash(openid),
        len(sources),
        concurrency,
        papers_dir,
    )

    sem = asyncio.Semaphore(concurrency)

    async def _guarded(src: str) -> tuple[str, bool, str]:
        async with sem:
            return await _ingest_one(cfg, openid, src)

    results = await asyncio.gather(*[_guarded(s) for s in sources])

    # 汇总统计
    ok = [r for r in results if r[1]]
    fail = [r for r in results if not r[1]]

    print(f"\n{'='*60}")
    print(f"批量导入完成：成功 {len(ok)} 条，失败 {len(fail)} 条")
    print(f"{'='*60}")

    if ok:
        print(f"\n✓ 成功（{len(ok)} 条）：")
        for src, _, msg in ok:
            short_src = src if len(src) <= 80 else src[:77] + "..."
            print(f"  [OK] {short_src}")
            print(f"       {msg}")

    if fail:
        print(f"\n✗ 失败（{len(fail)} 条）：")
        for src, _, msg in fail:
            short_src = src if len(src) <= 80 else src[:77] + "..."
            print(f"  [FAIL] {short_src}")
            print(f"         原因: {msg}")

    if fail:
        sys.exit(1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="批量向 RAG 数据库导入论文（URL 或本地文件）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--openid",
        required=True,
        help="目标用户的微信 openid（决定文献存入哪个用户目录）",
    )
    parser.add_argument(
        "sources",
        nargs="*",
        metavar="URL_OR_PATH",
        help="直接传入的 URL 或本地文件路径，可多个",
    )
    parser.add_argument(
        "--list",
        dest="list_file",
        metavar="FILE",
        help="文本文件路径，每行一条 URL 或文件路径（# 开头为注释）",
    )
    parser.add_argument(
        "--dir",
        dest="scan_dir",
        metavar="DIR",
        help="扫描指定目录下所有 PDF/HTML/TXT 文件并批量导入",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        metavar="N",
        help="同时并发的导入任务数（默认 3，避免触发外部限速）",
    )
    return parser.parse_args()


async def main() -> None:
    """解析命令行参数，收集来源，调用批量导入。"""
    args = _parse_args()

    cfg = AppConfig()

    # 收集所有来源
    all_sources: list[str] = list(args.sources or [])

    if args.list_file:
        all_sources.extend(_collect_from_list_file(Path(args.list_file)))

    if args.scan_dir:
        local_files = _collect_from_dir(Path(args.scan_dir))
        all_sources.extend(str(f) for f in local_files)

    if not all_sources:
        logger.error("未指定任何来源，请通过位置参数、--list 或 --dir 提供")
        sys.exit(1)

    # 去重（保留顺序）
    seen: set[str] = set()
    deduped: list[str] = []
    for s in all_sources:
        if s not in seen:
            seen.add(s)
            deduped.append(s)

    if len(deduped) < len(all_sources):
        logger.info("去重后剩余 %d 条（原 %d 条）", len(deduped), len(all_sources))

    await batch_ingest(cfg, args.openid, deduped, concurrency=args.concurrency)


if __name__ == "__main__":
    asyncio.run(main())
