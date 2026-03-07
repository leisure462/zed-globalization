#!/usr/bin/env python3
"""
编译前补丁脚本：修复 Agent 插件环境变量被覆盖问题。

补丁点 1: 删除强制清空 ANTHROPIC_API_KEY 的代码（兼容新旧两种位置）
补丁点 2: 透传 Claude Code 相关系统环境变量（兼容新旧两种文件）

用法: python3 patch_agent_env.py [--source-root zed] [--dry-run]
"""

import argparse
import io
import sys
from pathlib import Path

# Windows CI 默认 cp1252 编码，无法输出中文
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PATCH_MARKER = "[ZED_GLOBALIZATION_PATCH]"

# 需要透传的环境变量注入代码（Rust）
ENV_PASSTHROUGH_SNIPPET = """\
// {marker} 透传 Claude Code 相关系统环境变量
        for var_name in [
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_BASE_URL",
            "CLAUDE_CODE_USE_BEDROCK",
            "CLAUDE_CODE_USE_VERTEX",
        ] {{
            if let Ok(val) = std::env::var(var_name) {{
                extra_env.insert(var_name.into(), val);
            }}
        }}
        for (key, val) in std::env::vars() {{
            if key.starts_with("AWS_")
                || key.starts_with("GOOGLE_CLOUD_")
                || key == "CLOUD_ML_REGION"
            {{
                extra_env.insert(key, val);
            }}
        }}"""


def _read(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def _write(path: Path, content: str, dry_run: bool, name: str) -> None:
    if dry_run:
        print(f"  DRY-RUN: {name} 将被修改")
    else:
        path.write_text(content, encoding="utf-8")
        print(f"  OK: {name} 补丁成功")


def patch_remove_api_key_clear(source_root: Path, dry_run: bool) -> bool:
    """补丁点 1: 删除强制清空 ANTHROPIC_API_KEY 的代码行。

    旧版在 agent_server_store.rs，新版在 custom.rs (CLAUDE_AGENT_NAME 分支)。
    """
    candidates = [
        source_root / "crates/project/src/agent_server_store.rs",
        source_root / "crates/agent_servers/src/custom.rs",
    ]
    old_line = 'extra_env.insert("ANTHROPIC_API_KEY".into(), "".into());'
    # 旧版写法用 env 而非 extra_env
    old_line_legacy = 'env.insert("ANTHROPIC_API_KEY".into(), "".into());'

    for target in candidates:
        name = target.name
        content = _read(target)
        if content is None:
            continue
        if PATCH_MARKER in content and "已删除强制清空 ANTHROPIC_API_KEY" in content:
            print(f"  SKIP: {name} 已包含补丁标记，跳过")
            return True

        for needle in (old_line, old_line_legacy):
            if needle in content:
                replacement = f"// {PATCH_MARKER} 已删除强制清空 ANTHROPIC_API_KEY"
                patched = content.replace(needle, replacement, 1)
                _write(target, patched, dry_run, name)
                return True

    print("  WARN: 未找到强制清空 ANTHROPIC_API_KEY 的代码，上游可能已修改，跳过")
    return False


def patch_env_passthrough(source_root: Path, dry_run: bool) -> bool:
    """补丁点 2: 在 connect() 中透传系统环境变量。

    旧版 claude.rs + custom.rs 并存，新版仅 custom.rs。对所有匹配文件注入。
    """
    candidates = [
        source_root / "crates/agent_servers/src/claude.rs",
        source_root / "crates/agent_servers/src/custom.rs",
    ]
    anchor = "let mut extra_env = load_proxy_env(cx);"
    anchor_legacy = "let extra_env = load_proxy_env(cx);"
    patched_any = False

    for target in candidates:
        name = target.name
        content = _read(target)
        if content is None:
            continue
        if PATCH_MARKER in content and "透传 Claude Code" in content:
            print(f"  SKIP: {name} 已包含补丁标记，跳过")
            patched_any = True
            continue

        for needle in (anchor, anchor_legacy):
            if needle not in content:
                continue
            new_anchor = "let mut extra_env = load_proxy_env(cx);"
            inject = ENV_PASSTHROUGH_SNIPPET.format(marker=PATCH_MARKER)
            replacement = f"{new_anchor}\n{inject}"
            patched = content.replace(needle, replacement, 1)
            _write(target, patched, dry_run, name)
            patched_any = True
            break

    if not patched_any:
        print("  WARN: 未找到 load_proxy_env 调用，上游可能已修改，跳过")
    return patched_any


def main() -> int:
    parser = argparse.ArgumentParser(
        description="编译前补丁：修复 Agent 插件环境变量被覆盖问题"
    )
    parser.add_argument(
        "--source-root",
        default="zed",
        help="Zed 源码根目录（默认: zed）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅检查，不实际修改文件",
    )
    args = parser.parse_args()

    source_root = Path(args.source_root)
    if not source_root.is_dir():
        print(f"ERROR: 源码目录 {source_root} 不存在")
        return 1

    print(f"源码目录: {source_root.resolve()}")
    if args.dry_run:
        print("模式: dry-run（不修改文件）\n")
    else:
        print("模式: 正式补丁\n")

    print("[补丁 1] 删除强制清空 ANTHROPIC_API_KEY")
    r1 = patch_remove_api_key_clear(source_root, args.dry_run)

    print("[补丁 2] 透传 Claude Code 相关系统环境变量")
    r2 = patch_env_passthrough(source_root, args.dry_run)

    print()
    if r1 and r2:
        print("全部补丁已就绪。")
        return 0
    else:
        print("部分补丁未能应用，请检查上方 WARN 信息。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
