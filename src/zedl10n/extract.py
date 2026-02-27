"""字符串提取 + 上下文收集"""

from __future__ import annotations

import argparse
import logging
import re
import string
from collections import Counter
from pathlib import Path

from .utils import ContextDict, TranslationDict, save_json

log = logging.getLogger(__name__)

# 匹配双引号字符串（处理转义）
_STRING_PATTERN = re.compile(r'"((?:\\.|[^"\\])*)"')

# 排除包含 json_path: 的行
_JSON_PATH_PATTERN = re.compile(r'json_path:\s*".*?"')
_URL_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")
_PATH_PREFIX_PATTERN = re.compile(r"^(?:[A-Za-z]:[\\/]|/|\./|\.\./)")
_HASH_PATTERN = re.compile(r"^[0-9a-fA-F]{8,}$")
_VERSION_PATTERN = re.compile(r"^v?\d+(?:\.\d+){1,3}(?:[-._][A-Za-z0-9]+)?$")
_PLACEHOLDER_ONLY_PATTERN = re.compile(
    r"^(?:\s*(?:\{[^{}]*\}|%[-+#0-9.]*[a-zA-Z])\s*)+$",
)
_IDENTIFIER_PATTERN = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)*$",
)


def _is_translatable_string(s: str, line: str = "") -> tuple[bool, str]:
    """过滤明显非 UI 字符串，返回 (是否保留, 原因)"""
    text = s.strip()
    if not text:
        return False, "empty"

    if all(ch in string.punctuation for ch in text):
        return False, "punctuation_only"

    if _URL_PATTERN.match(text):
        return False, "url"

    if _PATH_PREFIX_PATTERN.match(text):
        return False, "path_like"

    if "/" in text and " " not in text and text.count("/") >= 2:
        return False, "path_like"

    if text.startswith(("--", "-")) and " " not in text:
        return False, "cli_flag"

    if _PLACEHOLDER_ONLY_PATTERN.fullmatch(text):
        return False, "placeholder_only"

    if _HASH_PATTERN.fullmatch(text):
        return False, "hash_like"

    if _VERSION_PATTERN.fullmatch(text):
        return False, "version_like"

    if _IDENTIFIER_PATTERN.fullmatch(text) and ("_" in text or "::" in text):
        return False, "code_identifier"

    if text.isupper() and "_" in text:
        return False, "const_like"

    if "." in text and " " not in text and text.split(".")[-1] in {
        "rs", "toml", "json", "yaml", "yml", "md", "txt", "lock", "dll", "exe",
        "so", "dylib", "png", "jpg", "svg",
    }:
        return False, "file_like"

    if (
        "::" in line
        and " " not in text
        and any(token in line for token in ("match ", "=>", "enum ", "struct "))
    ):
        return False, "code_context"

    return True, "keep"


def extract_strings(content: str) -> list[str]:
    """从文件内容中提取所有双引号字符串"""
    filtered = _JSON_PATH_PATTERN.sub("", content)
    return _STRING_PATTERN.findall(filtered)


def extract_with_context(
    content: str, context_lines: int = 3
) -> tuple[list[str], ContextDict]:
    """提取字符串并收集 +-N 行代码上下文"""
    filtered = _JSON_PATH_PATTERN.sub("", content)
    lines = filtered.splitlines()
    strings: list[str] = []
    contexts: ContextDict = {}
    skipped_reasons: Counter[str] = Counter()

    for i, line in enumerate(lines):
        for match in _STRING_PATTERN.finditer(line):
            s = match.group(1)
            keep, reason = _is_translatable_string(s, line)
            if not keep:
                skipped_reasons[reason] += 1
                continue
            strings.append(s)
            start = max(0, i - context_lines)
            end = min(len(lines), i + context_lines + 1)
            ctx_block = "\n".join(lines[start:end])
            contexts[s] = {"line": i + 1, "context": ctx_block}

    if skipped_reasons:
        total = sum(skipped_reasons.values())
        top_reasons = ", ".join(
            f"{reason}={count}"
            for reason, count in skipped_reasons.most_common(4)
        )
        log.debug("过滤非 UI 字符串 %d 条 (%s)", total, top_reasons)

    return strings, contexts


def extract_all(
    file_paths: list[str],
    output_path: str = "string.json",
    context_path: str = "string_context.json",
) -> TranslationDict:
    """从文件列表中提取字符串，输出 string.json 和 context.json"""
    all_strings: TranslationDict = {}
    all_contexts: dict[str, ContextDict] = {}

    skipped: list[str] = []
    for fp_str in file_paths:
        fp = Path(fp_str)
        if not fp.exists():
            log.warning("文件不存在，跳过: %s", fp)
            skipped.append(fp_str)
            continue

        try:
            content = fp.read_text(encoding="utf-8", errors="ignore")
        except OSError as e:
            log.warning("读取失败 %s: %s", fp, e)
            skipped.append(fp_str)
            continue

        strings, contexts = extract_with_context(content)
        if strings:
            key = str(fp).replace("\\", "/")
            all_strings[key] = {s: "" for s in strings}
            all_contexts[key] = contexts
            log.debug("提取 %d 条: %s", len(strings), key)
        else:
            log.warning("未提取到字符串，跳过: %s", fp)
            skipped.append(fp_str)

    if skipped:
        log.warning(
            "共 %d 个文件被跳过（输入 %d，提取 %d）:",
            len(skipped), len(file_paths), len(all_strings),
        )
        for s in skipped:
            log.warning("  - %s", s)

    save_json(all_strings, output_path)
    save_json(all_contexts, context_path)
    log.info(
        "提取完成: %d 个文件, 输出 %s 和 %s",
        len(all_strings),
        output_path,
        context_path,
    )
    return all_strings


def run(args: argparse.Namespace) -> None:
    """CLI 入口"""
    if args.files:
        file_paths = args.files
    else:
        # 没有指定文件列表时，使用 scan 自动扫描
        log.info("未指定文件列表，正在使用 AI 扫描...")
        from .scan import find_all_rs_files

        root = Path(args.source_root)
        file_paths = [str(f) for f in find_all_rs_files(root)]
        log.info("使用全量 .rs 文件列表（%d 个文件）", len(file_paths))

    extract_all(file_paths, args.output, "string_context.json")
