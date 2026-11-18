"""把核电语料归档成版本化快照（数据层，git 之外）。

背景：``document_agent/`` 下的语料会被整体更换（"目前的核电语料我都会更换的"）。
下游三件事——规程规则抽取、规则表槽位查询接口、灌 LightRAG——都必须能回答
"这份结论是哪一版语料得出的"。所以语料本身需要一层**独立的存档**，而不是
直接读用户当前盘上的文件。

按项目的三层分离（数据 / 配置 / 代码）：

* 数据：``document_agent/corpus/``（不在任何 git 仓库内）
* 配置：下面的 ``SOURCE_GROUPS`` 与环境变量
* 代码：本脚本（在 git 仓库内）

存档目录结构::

    corpus/
      manifest.json            当前快照的清单
      manifests/<时间戳>.json  历史：每次建档留一份，用于比对换版
      files/<相对路径>          原件副本（逐字节一致，保留 mtime）
      parsed/<文件名>.parsed/   LightRAG native 解析产物（blocks/tables/drawings/图片）
      text/<文件名>.txt         解析产物投影出的纯文本（表格转 markdown）
      README.md                自动生成的存档概览

用法::

    python scripts/corpus_archive.py build     # 建档（增量：内容没变就跳过重解析）
    python scripts/corpus_archive.py verify    # 核对当前源目录与存档的差异
    python scripts/corpus_archive.py status    # 打印存档概览

环境变量（都有默认值，按需覆盖）::

    CORPUS_SOURCE_ROOT      语料根目录，默认是本仓库的上级目录 document_agent/
    CORPUS_ARCHIVE_ROOT     存档根目录，默认 <source_root>/corpus
    CORPUS_LIGHTRAG_PYTHON  LightRAG 解释器（用于离线解析 docx）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# 语料分组：只归档这三组，其余目录（三个 git 仓库、.env）不碰。
SOURCE_GROUPS = ("设备操作规程", "标准文件数据集", "示例文档及图纸")

# 文件名形如 "G-FH01-VMP-1501 装卸料机操作 R6.docx"：编号 + 名称 + 版次 + 状态标记。
RE_DOC_CODE = re.compile(r"^(?P<code>G-[A-Z0-9]+(?:-[A-Z0-9]+)*)")
RE_REVISION = re.compile(r"(?P<rev>R\d+(?:-[一-鿿]+)?)")
RE_STD_CODE = re.compile(r"^(?P<std>[A-Za-z]{1,4}[/-]?T?[ ]?\d{3,5}(?:\.\d+)?-\d{4})")
RE_COPY_MARK = re.compile(r"\((?P<n>\d+)\)\s*$")
# 状态标记：归档版 / 上传版 / 初版 / 归档 …
STATUS_WORDS = ("归档版", "上传版", "初版", "归档", "作废")

TABLE_TAG_RE = re.compile(
    r"<table\s+id=\"(?P<id>[^\"]+)\"\s+format=\"json\">(?P<body>.*?)</table>",
    re.DOTALL,
)
DRAWING_RE = re.compile(r"<drawing[^>]*/>")


# ---------------------------------------------------------------- 路径解析
def _default_source_root() -> Path:
    """语料根目录：默认取本仓库的上级目录（document_agent/）。"""
    env = os.getenv("CORPUS_SOURCE_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    # scripts/ -> local-deep-research/ -> document_agent/
    return Path(__file__).resolve().parents[2]


def _default_archive_root(source_root: Path) -> Path:
    env = os.getenv("CORPUS_ARCHIVE_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return source_root / "corpus"


def _lightrag_python(source_root: Path) -> Path | None:
    """LightRAG 解释器：用于离线跑它的 native 解析器（不需要起服务）。"""
    env = os.getenv("CORPUS_LIGHTRAG_PYTHON")
    if env:
        path = Path(env).expanduser()
        return path if path.exists() else None
    for rel in (
        Path("LightRAG/.venv/Scripts/python.exe"),
        Path("LightRAG/.venv/bin/python"),
        Path("LightRAG/.venv/Scripts/python"),
    ):
        candidate = source_root / rel
        if candidate.exists():
            return candidate
    return None


# ---------------------------------------------------------------- 文件名解析
def parse_filename(stem: str) -> dict[str, Any]:
    """从文件名拆出编号 / 名称 / 版次 / 状态标记。

    拆不干净不算失败——规程的编号体例本来就不统一
    （"G-AA装换料操作规程" 没有版次，"G-MZ13-VMP-1501阻力塞抽插换位操作 R7" 编号后
    没有空格），所以每个字段都允许为空，原始文件名始终保留在清单里。
    """
    info: dict[str, Any] = {
        "doc_code": "",
        "std_code": "",
        "title": stem.strip(),
        "revision": "",
        "status_tags": [],
    }

    std = RE_STD_CODE.match(stem)
    if std:
        info["std_code"] = std.group("std")
        info["title"] = stem[std.end():].strip() or stem.strip()
        return info

    rest = stem
    code = RE_DOC_CODE.match(stem)
    if code:
        info["doc_code"] = code.group("code")
        rest = stem[code.end():].strip()

    copy_mark = RE_COPY_MARK.search(rest)
    if copy_mark:
        info["status_tags"].append(f"副本{copy_mark.group('n')}")
        rest = rest[: copy_mark.start()].strip()

    revision = RE_REVISION.search(rest)
    if revision:
        info["revision"] = revision.group("rev")
        rest = (rest[: revision.start()] + " " + rest[revision.end():]).strip()

    tags = [word for word in STATUS_WORDS if word in rest]
    info["status_tags"].extend(tags)
    for word in tags:
        rest = rest.replace(word, "")
    info["title"] = re.sub(r"\s{2,}", " ", rest).strip(" -_、,") or stem.strip()
    return info


# ---------------------------------------------------------------- 基础工具
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_manifest(archive_root: Path) -> dict[str, Any]:
    path = archive_root / "manifest.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


# ---------------------------------------------------------------- 文本投影
def _render_table(body: str) -> str:
    """把 ``<table>`` 里的二维数组渲染成 markdown 表格。"""
    try:
        rows = json.loads(body)
    except json.JSONDecodeError:
        return ""
    if not isinstance(rows, list) or not rows:
        return ""
    lines = []
    for row in rows:
        if not isinstance(row, list):
            row = [row]
        cells = [DRAWING_RE.sub("[图片]", str(cell)).strip() for cell in row]
        lines.append("| " + " | ".join(cells) + " |")
    if len(lines) > 1:
        lines.insert(1, "| " + " | ".join(["---"] * len(rows[0])) + " |")
    return "\n".join(lines)


def project_text(blocks_path: Path) -> tuple[str, dict[str, Any]]:
    """把 blocks.jsonl 投影成可读纯文本。

    表格在 block 内容里是 ``<table id=... format="json">[[...]]</table>`` 的形式，
    直接留原文对人和检索都不友好——投影时换成 markdown 表格。
    """
    stats = {"blocks": 0, "tables": 0, "chars": 0}
    chunks: list[str] = []
    for raw in blocks_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            block = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if block.get("type") == "meta":
            continue
        content = block.get("content") or ""
        heading = block.get("heading") or ""
        parents = block.get("parent_headings") or []

        def _sub(match: re.Match[str]) -> str:
            stats["tables"] += 1
            return _render_table(match.group("body"))

        content = TABLE_TAG_RE.sub(_sub, content)
        if not content.strip() and not heading:
            continue
        stats["blocks"] += 1
        trail = " > ".join([str(p) for p in parents] + ([heading] if heading else []))
        chunks.append(f"## {trail}\n{content.strip()}" if trail else content.strip())

    text = "\n\n".join(chunks).strip() + "\n"
    stats["chars"] = len(text)
    return text, stats


# ---------------------------------------------------------------- 解析/探测
def parse_docx(
    python_exe: Path,
    lightrag_root: Path,
    source: Path,
    out_dir: Path,
) -> dict[str, Any]:
    """用 LightRAG 的 native 解析器离线解析 docx。

    走子进程而不是 import：解析器住在另一个仓库里，子进程边界能把两个仓库
    解耦（LightRAG 换个 venv、换个分支都不影响本脚本）。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(python_exe),
        "-m",
        "lightrag.parser.cli",
        str(source),
        "--engine",
        "native",
        "-o",
        str(out_dir),
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(lightrag_root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=600,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
        return {"status": "failed", "error": " | ".join(tail)}
    return {"status": "parsed"}


def probe_pdf(path: Path, max_pages: int = 3) -> dict[str, Any]:
    """探测 PDF：页数 + 有无文本层。

    标准文件数据集那三份 PDF 是扫描件（每页 0 字符），没有文本层就进不了
    检索层，只能等 OCR——这个事实必须写进清单，不能靠"入库成功"来蒙。
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        return {"status": "probe_failed", "error": "pypdf 不可用"}
    try:
        reader = PdfReader(str(path))
        pages = len(reader.pages)
        chars = 0
        for index in range(min(max_pages, pages)):
            try:
                chars += len(reader.pages[index].extract_text() or "")
            except Exception:  # noqa: BLE001 - 单页坏了不影响整体判断
                continue
    except Exception as exc:  # noqa: BLE001
        return {"status": "probe_failed", "error": f"{type(exc).__name__}: {exc}"}
    status = "has_text_layer" if chars > 0 else "no_text_layer"
    return {
        "status": status,
        "pages": pages,
        "probed_pages": min(max_pages, pages),
        "text_chars": chars,
    }


# ---------------------------------------------------------------- 建档
def _copy_if_changed(source: Path, target: Path, digest: str) -> str:
    """复制到存档；内容没变就跳过（幂等，重跑不产生无意义写入）。"""
    if target.exists() and sha256_file(target) == digest:
        return "unchanged"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return "copied"


def _purge_temp_copies(archive_root: Path) -> list[str]:
    """清掉存档里残留的临时文件副本（Word 锁文件等）。

    只删 ``~$`` / ``.~lock.`` 开头的文件——真实语料的副本**永不自动删除**：
    语料换版时旧快照正是要留下来的东西，删了就失去存档的意义。
    """
    removed: list[str] = []
    files_root = archive_root / "files"
    if not files_root.is_dir():
        return removed
    for path in files_root.rglob("*"):
        if not path.is_file():
            continue
        if not (path.name.startswith("~$") or path.name.startswith(".~lock.")):
            continue
        parsed_dir = archive_root / "parsed" / f"{path.name}.parsed"
        shutil.rmtree(parsed_dir, ignore_errors=True)
        path.unlink()
        removed.append(path.relative_to(archive_root).as_posix())
    return removed


def build(args: argparse.Namespace) -> int:
    source_root = Path(args.source_root).expanduser().resolve()
    archive_root = Path(args.archive_root).expanduser().resolve()
    groups = tuple(args.groups) if args.groups else SOURCE_GROUPS

    if not source_root.is_dir():
        print(f"[x] 语料根目录不存在：{source_root}", file=sys.stderr)
        return 2

    python_exe = _lightrag_python(source_root)
    lightrag_root = source_root / "LightRAG"
    if python_exe is None or not lightrag_root.is_dir():
        print("[!] 找不到 LightRAG 解析器，docx 只登记不解析", file=sys.stderr)

    previous = {
        entry["relpath"]: entry
        for entry in (_load_manifest(archive_root).get("files") or [])
    }

    entries: list[dict[str, Any]] = []
    for group in groups:
        group_dir = source_root / group
        if not group_dir.is_dir():
            print(f"[!] 分组不存在，跳过：{group}", file=sys.stderr)
            continue
        for path in sorted(group_dir.rglob("*")):
            if not path.is_file():
                continue
            if path.name.startswith("~$") or path.name.startswith(".~lock."):
                # Word / LibreOffice 的锁文件与临时文件，不是语料。
                print(f"  [     跳过] {path.relative_to(source_root).as_posix()}（临时文件）")
                continue
            relpath = path.relative_to(source_root).as_posix()
            digest = sha256_file(path)
            meta = parse_filename(path.stem)

            entry: dict[str, Any] = {
                "relpath": relpath,
                "group": group,
                "filename": path.name,
                "ext": path.suffix.lower(),
                "bytes": path.stat().st_size,
                "sha256": digest,
                "source_mtime": datetime.fromtimestamp(
                    path.stat().st_mtime, timezone.utc
                ).isoformat(timespec="seconds"),
                **meta,
            }

            prior = previous.get(relpath)
            unchanged = bool(prior) and prior.get("sha256") == digest

            # 原件副本
            copy_state = _copy_if_changed(
                path, archive_root / "files" / relpath, digest
            )
            entry["archived_file"] = f"files/{relpath}"
            entry["copy_state"] = copy_state

            # 解析 / 探测
            if path.suffix.lower() == ".docx" and python_exe is not None:
                # 解析器的命名规则：目录用**完整文件名**（含扩展名），里面的
                # blocks 文件用不带扩展名的名字——两处都得对上，否则会误判成
                # "解析成功但产物不存在"。
                parsed_dir = archive_root / "parsed" / f"{path.name}.parsed"
                blocks_file = parsed_dir / f"{path.stem}.blocks.jsonl"
                text_path = archive_root / "text" / f"{path.stem}.txt"
                if unchanged and blocks_file.exists():
                    # 产物还在就不重解析；但文本投影可能缺（上次路径写错过），
                    # 缺了就补投影——重解析一份 10MB 的 docx 很贵。
                    if not text_path.exists():
                        text, stats = project_text(blocks_file)
                        text_path.parent.mkdir(parents=True, exist_ok=True)
                        text_path.write_text(text, encoding="utf-8")
                    entry["parse"] = {
                        **(prior.get("parse") or {}),
                        "status": "parsed",
                        "reused": True,
                        "parsed_dir": f"parsed/{path.name}.parsed",
                        "text_path": f"text/{path.stem}.txt",
                    }
                else:
                    outcome = parse_docx(
                        python_exe, lightrag_root, path, archive_root / "parsed"
                    )
                    if outcome.get("status") == "parsed" and blocks_file.exists():
                        text, stats = project_text(blocks_file)
                        text_path.parent.mkdir(parents=True, exist_ok=True)
                        text_path.write_text(text, encoding="utf-8")
                        entry["parse"] = {
                            **outcome,
                            **stats,
                            "parsed_dir": f"parsed/{path.name}.parsed",
                            "text_path": f"text/{path.stem}.txt",
                        }
                    else:
                        entry["parse"] = outcome
            elif path.suffix.lower() == ".pdf":
                entry["parse"] = probe_pdf(path)
            else:
                entry["parse"] = {"status": "skipped", "reason": "不支持的格式"}

            entries.append(entry)
            flag = {
                "parsed": "解析",
                "failed": "失败",
                "has_text_layer": "有文本层",
                "no_text_layer": "无文本层",
                "skipped": "跳过",
                "probe_failed": "探测失败",
            }.get(entry["parse"].get("status"), entry["parse"].get("status"))
            print(f"  [{flag:>10}] {relpath}")

    manifest = {
        "schema": 1,
        "created_at": _now_iso(),
        "source_root": str(source_root),
        "archive_root": str(archive_root),
        "groups": list(groups),
        "totals": {
            "files": len(entries),
            "bytes": sum(e["bytes"] for e in entries),
            "parsed": sum(1 for e in entries if e["parse"].get("status") == "parsed"),
            "no_text_layer": sum(
                1 for e in entries if e["parse"].get("status") == "no_text_layer"
            ),
        },
        "packages": _scan_packages(source_root),
        "files": entries,
    }

    archive_root.mkdir(parents=True, exist_ok=True)
    for stale in _purge_temp_copies(archive_root):
        print(f"  [     清理] {stale}（临时文件残留）")
    (archive_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    history = archive_root / "manifests" / f"{stamp}.json"
    history.parent.mkdir(parents=True, exist_ok=True)
    history.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (archive_root / "README.md").write_text(
        _render_readme(manifest), encoding="utf-8"
    )

    totals = manifest["totals"]
    print(
        f"\n建档完成：{totals['files']} 份文件 / {totals['bytes'] / 1024 / 1024:.1f} MB"
        f"（解析 {totals['parsed']}，扫描件 {totals['no_text_layer']}）"
        f"\n  清单：{archive_root / 'manifest.json'}"
    )
    return 0


def _scan_packages(source_root: Path) -> list[dict[str, Any]]:
    """记录语料根目录下的交付压缩包（原件清单 + 指纹），不复制。"""
    packages: list[dict[str, Any]] = []
    for zip_path in sorted(source_root.glob("*.zip")):
        try:
            with zipfile.ZipFile(zip_path) as archive:
                names = archive.namelist()
        except (OSError, zipfile.BadZipFile) as exc:
            packages.append(
                {"filename": zip_path.name, "error": f"{type(exc).__name__}: {exc}"}
            )
            continue
        packages.append(
            {
                "filename": zip_path.name,
                "bytes": zip_path.stat().st_size,
                "sha256": sha256_file(zip_path),
                "entries": len(names),
                "members": names,
            }
        )
    return packages


def _render_readme(manifest: dict[str, Any]) -> str:
    totals = manifest["totals"]
    lines = [
        "# 核电语料存档",
        "",
        f"- 建档时间：{manifest['created_at']}",
        f"- 语料根目录：`{manifest['source_root']}`",
        f"- 规模：{totals['files']} 份文件 / {totals['bytes'] / 1024 / 1024:.1f} MB",
        "",
        "> 本目录是 git 之外的**数据层**。语料换版时，`manifests/` 里会留下每次建档的",
        "> 清单快照，用 `corpus_archive.py verify` 比对新增 / 变更 / 缺失。",
        "",
        "## 分组",
        "",
        "| 分组 | 文件数 | 说明 |",
        "| --- | --- | --- |",
    ]
    by_group: dict[str, list[dict[str, Any]]] = {}
    for entry in manifest["files"]:
        by_group.setdefault(entry["group"], []).append(entry)
    notes = {
        "设备操作规程": "设备操作规程（docx，含表格与图片）",
        "标准文件数据集": "国家/行业标准（PDF，多为扫描件）",
        "示例文档及图纸": "示例模型图纸与设计准则",
    }
    for group, items in by_group.items():
        lines.append(f"| {group} | {len(items)} | {notes.get(group, '')} |")

    lines += [
        "",
        "## 文件清单",
        "",
        "| 文件 | 编号 | 版次 | 解析状态 |",
        "| --- | --- | --- | --- |",
    ]
    for entry in manifest["files"]:
        status = {
            "parsed": "已解析",
            "failed": "解析失败",
            "has_text_layer": "有文本层",
            "no_text_layer": "无文本层（扫描件，待 OCR）",
            "skipped": "跳过",
            "probe_failed": "探测失败",
        }.get(entry["parse"].get("status"), entry["parse"].get("status", ""))
        lines.append(
            f"| {entry['filename']} | "
            f"{entry.get('doc_code') or entry.get('std_code') or '—'} | "
            f"{entry.get('revision') or '—'} | {status} |"
        )

    blocked = [
        e for e in manifest["files"] if e["parse"].get("status") == "no_text_layer"
    ]
    if blocked:
        lines += [
            "",
            "## 待办：扫描件无文本层",
            "",
            "以下 PDF 逐页取不到文字，**进不了检索层**，需要 OCR 后才能被问答引用：",
            "",
        ]
        for entry in blocked:
            lines.append(
                f"- `{entry['filename']}`（{entry['parse'].get('pages', '?')} 页）"
            )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- 核对
def verify(args: argparse.Namespace) -> int:
    """比对当前源目录与存档清单：新增 / 变更 / 缺失。"""
    source_root = Path(args.source_root).expanduser().resolve()
    archive_root = Path(args.archive_root).expanduser().resolve()
    manifest = _load_manifest(archive_root)
    if not manifest:
        print("[x] 没有存档清单，先跑 build", file=sys.stderr)
        return 2

    groups = tuple(manifest.get("groups") or SOURCE_GROUPS)
    current: dict[str, Path] = {}
    for group in groups:
        group_dir = source_root / group
        if not group_dir.is_dir():
            continue
        for path in sorted(group_dir.rglob("*")):
            if path.is_file() and not path.name.startswith("~$"):
                current[path.relative_to(source_root).as_posix()] = path

    archived = {e["relpath"]: e for e in manifest.get("files") or []}

    added = sorted(set(current) - set(archived))
    removed = sorted(set(archived) - set(current))
    changed = []
    for relpath in sorted(set(current) & set(archived)):
        if sha256_file(current[relpath]) != archived[relpath].get("sha256"):
            changed.append(relpath)

    print(f"存档快照：{manifest.get('created_at')}")
    print(f"新增 {len(added)} 份 / 变更 {len(changed)} 份 / 源里已删 {len(removed)} 份")
    for relpath in added:
        print(f"  [+ 新增] {relpath}")
    for relpath in changed:
        print(f"  [~ 变更] {relpath}")
    for relpath in removed:
        print(f"  [- 缺失] {relpath}")
    if not (added or changed or removed):
        print("  源目录与存档一致")
    return 1 if (added or changed or removed) else 0


# ---------------------------------------------------------------- 概览
def _by_group(manifest: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in manifest.get("files") or []:
        grouped.setdefault(entry.get("group", "?"), []).append(entry)
    return grouped


def status(args: argparse.Namespace) -> int:
    archive_root = Path(args.archive_root).expanduser().resolve()
    manifest = _load_manifest(archive_root)
    if not manifest:
        print("[x] 没有存档清单，先跑 build", file=sys.stderr)
        return 2
    totals = manifest["totals"]
    print(f"快照时间：{manifest.get('created_at')}")
    print(f"规模：{totals['files']} 份 / {totals['bytes'] / 1024 / 1024:.1f} MB")
    print(f"已解析 {totals['parsed']} 份，无文本层 {totals['no_text_layer']} 份")
    for group, items in _by_group(manifest).items():
        print(f"\n[{group}] {len(items)} 份")
        for entry in items:
            print(
                f"  {entry['filename'][:52]:54s} "
                f"{entry['parse'].get('status', '?'):16s} "
                f"{entry.get('doc_code') or entry.get('std_code') or ''}"
            )
    return 0


# ---------------------------------------------------------------- 入口
def main(argv: list[str] | None = None) -> int:
    source_root = _default_source_root()
    parser = argparse.ArgumentParser(
        prog="corpus_archive",
        description="核电语料归档：build 建档 / verify 核对 / status 概览",
    )
    parser.add_argument(
        "command", choices=("build", "verify", "status"), nargs="?", default="build"
    )
    parser.add_argument("--source-root", default=str(source_root))
    parser.add_argument(
        "--archive-root", default=str(_default_archive_root(source_root))
    )
    parser.add_argument(
        "--groups", nargs="*", default=list(SOURCE_GROUPS), help="要归档的分组目录名"
    )
    args = parser.parse_args(argv)
    if args.command == "build":
        return build(args)
    if args.command == "verify":
        return verify(args)
    return status(args)


if __name__ == "__main__":
    raise SystemExit(main())
