"""AI 并发翻译，内置智能过滤"""

from __future__ import annotations

import argparse
import asyncio
import logging
import random
from collections import defaultdict
from pathlib import Path

from .batch import split_batch
from .prompts import (
    SYSTEM_PROMPT_TEMPLATE,
    XML_FALLBACK_INSTRUCTION,
    build_consistency_fix_prompt,
    build_fix_prompt,
    build_numbered_instruction,
    build_user_prompt,
    validate_placeholders,
)
from .utils import (
    AIConfig,
    ProgressBar,
    TranslationDict,
    build_glossary_section,
    load_json,
    normalize_fullwidth,
    parse_json_response,
    parse_numbered_response,
    parse_xml_response,
    save_json,
)

log = logging.getLogger(__name__)


def _build_translation_memory(data: TranslationDict) -> dict[str, str]:
    """从已有翻译构建全局记忆库（原文 -> 译文）。"""
    memory: dict[str, str] = {}
    for pairs in data.values():
        for original, translated in pairs.items():
            if original and translated:
                memory[original] = translated
    return memory


async def _call_ai(
    client: object,
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> str:
    """调用 AI API，内置网络错误重试"""
    for attempt in range(5):
        try:
            response = await client.chat.completions.create(  # type: ignore[attr-defined]
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                max_tokens=65535,
                extra_body={"thinking": {"type": "disabled"}},
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            if attempt < 4:
                delay = (2**attempt) * 3 + random.uniform(0, 2)
                log.debug("网络错误，等待 %.1fs 重试 (%d/5)", delay, attempt + 1)
                await asyncio.sleep(delay)
                continue
            raise
    return ""


async def _fetch_translation(
    client: object,
    model: str,
    file_path: str,
    strings: dict[str, str],
    file_content: str,
    system_prompt: str,
) -> tuple[dict[str, str], str | None]:
    """通过 JSON → XML(CDATA) → 编号格式 三级降级获取翻译结果"""
    user_prompt = build_user_prompt(file_path, strings, file_content)
    parse_failed = {"json": 0, "xml": 0, "numbered": 0}
    last_error = ""

    # 第一级: JSON 格式重试 3 次
    for attempt in range(3):
        try:
            raw = await _call_ai(client, model, system_prompt, user_prompt)
            result = parse_json_response(raw)
            if result:
                return result, None
            parse_failed["json"] += 1
        except Exception as e:
            last_error = f"JSON 请求失败: {type(e).__name__}: {e}"
            log.warning("翻译失败 %s [JSON %d/3]: %s", file_path, attempt + 1, e)

    # 第二级: XML(CDATA) 格式重试 3 次
    xml_prompt = user_prompt + XML_FALLBACK_INSTRUCTION
    for attempt in range(3):
        try:
            raw = await _call_ai(client, model, system_prompt, xml_prompt)
            result = parse_xml_response(raw)
            if result:
                return result, None
            parse_failed["xml"] += 1
            log.debug("XML 解析重试 (%d/3): %s", attempt + 1, file_path)
        except Exception as e:
            last_error = f"XML 请求失败: {type(e).__name__}: {e}"
            log.warning("翻译失败 %s [XML %d/3]: %s", file_path, attempt + 1, e)

    # 第三级: 编号格式重试 3 次
    keys = list(strings.keys())
    numbered_prompt = user_prompt + build_numbered_instruction(len(keys))
    for attempt in range(3):
        try:
            raw = await _call_ai(client, model, system_prompt, numbered_prompt)
            result = parse_numbered_response(raw, keys)
            if result:
                return result, None
            parse_failed["numbered"] += 1
            log.debug("编号格式解析重试 (%d/3): %s", attempt + 1, file_path)
        except Exception as e:
            last_error = f"编号请求失败: {type(e).__name__}: {e}"
            log.warning("翻译失败 %s [编号 %d/3]: %s", file_path, attempt + 1, e)

    reason = (
        "JSON 解析失败 {json} 次, XML 解析失败 {xml} 次, "
        "编号解析失败 {numbered} 次"
    ).format(**parse_failed)
    if last_error:
        reason = f"{reason}; 最后错误: {last_error}"

    log.warning(
        "[FAILED] JSON+XML+编号 均失败: %s (%d 条) | %s",
        file_path,
        len(strings),
        reason,
    )
    return {}, reason


async def _translate_batch(
    client: object,
    model: str,
    file_path: str,
    strings: dict[str, str],
    file_content: str,
    system_prompt: str,
) -> tuple[dict[str, str], str | None]:
    """翻译一批字符串，含占位符校验和自动重试"""
    result, failure_reason = await _fetch_translation(
        client, model, file_path, strings, file_content, system_prompt,
    )
    if not result:
        return result, failure_reason or "未返回可解析译文"

    # 占位符校验 + 重试（最多 2 次）
    for retry in range(2):
        errors = validate_placeholders(result)
        if not errors:
            return result, None
        log.debug(
            "占位符不匹配 %d 条，重试修正 (%d/2): %s",
            len(errors), retry + 1, file_path,
        )
        fix_prompt = build_fix_prompt(errors, result)
        try:
            raw = await _call_ai(client, model, system_prompt, fix_prompt)
            fixed = parse_json_response(raw)
        except Exception as e:
            log.warning("占位符修正请求失败 %s: %s", file_path, e)
            break
        if fixed:
            for key, val in fixed.items():
                if key in result:
                    result[key] = val

    # 最终校验：仍有问题的条目丢弃为空字符串
    final_errors = validate_placeholders(result)
    for original, (src_ph, dst_ph) in final_errors.items():
        log.warning(
            "占位符校验失败，丢弃译文: %r (原文占位符=%s, 译文占位符=%s) [%s]",
            original, src_ph, dst_ph, file_path,
        )
        result[original] = ""

    return result, None


def _read_source_file(file_path: str, source_root: str) -> str:
    """读取源文件内容，找不到则返回空字符串"""
    if not source_root:
        return ""
    root = Path(source_root)
    candidates = [root / file_path]
    if file_path.startswith("zed/"):
        candidates.append(root / file_path[4:])
    for p in candidates:
        if p.exists():
            try:
                return p.read_text(encoding="utf-8")
            except Exception:
                return ""
    return ""


async def _ai_fix_consistency(
    client: object,
    model: str,
    system_prompt: str,
    result: TranslationDict,
    glossary_path: str,
) -> tuple[TranslationDict, list[str]]:
    """用 AI 修复一致性问题，返回 (修复后结果, 修复日志)"""
    from .consistency import build_issues_for_ai, check_consistency

    issues = check_consistency(result, glossary_path)
    if not issues:
        return result, []

    log.info("一致性检查发现 %d 个问题，调用 AI 修复", len(issues))
    incon, glossary_v, keep_v = build_issues_for_ai(issues, result)
    if not incon and not glossary_v and not keep_v:
        return result, []

    user_prompt = build_consistency_fix_prompt(incon, glossary_v, keep_v)

    fix_log: list[str] = []
    try:
        raw = await _call_ai(client, model, system_prompt, user_prompt)
        fixed = parse_json_response(raw)
    except Exception as e:
        log.warning("AI 一致性修复请求失败: %s", e)
        return result, fix_log

    if not fixed:
        log.warning("AI 一致性修复返回为空")
        return result, fix_log

    # 将 AI 修正结果应用到所有文件
    for original, new_translation in fixed.items():
        if not new_translation:
            continue
        applied = 0
        for pairs in result.values():
            if original in pairs and pairs[original]:
                if pairs[original] != new_translation:
                    pairs[original] = new_translation
                    applied += 1
        if applied:
            fix_log.append(
                f'AI 修复: "{original}" → "{new_translation}" '
                f"(更新 {applied} 处)",
            )

    return result, fix_log


async def _translate_async(
    all_strings: TranslationDict,
    existing: TranslationDict,
    mode: str,
    lang: str,
    glossary_path: str,
    ai_cfg: AIConfig,
    source_root: str = "",
) -> tuple[TranslationDict, list[dict[str, str | int]], int]:
    """异步并发翻译"""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=ai_cfg.base_url, api_key=ai_cfg.api_key)
    semaphore = asyncio.Semaphore(ai_cfg.concurrency)

    glossary_section = build_glossary_section(glossary_path)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        lang=lang,
        glossary_section=glossary_section,
    )

    result: TranslationDict = {fp: dict(v) for fp, v in existing.items()}
    translation_memory = _build_translation_memory(existing) if mode != "full" else {}
    source_locations: dict[str, set[str]] = defaultdict(set)
    source_owner: dict[str, str] = {}
    reused_from_memory = 0

    # 先做全局去重：同一原文只翻译一次，然后分发到所有文件
    for file_path, strings in all_strings.items():
        result.setdefault(file_path, {})
        current = result[file_path]
        for s in strings:
            # incremental 模式下：空译文也会进入翻译，避免历史失败条目长期漏翻
            has_value = bool(current.get(s))
            need_translate = mode == "full" or not has_value
            if not need_translate:
                if s not in translation_memory and current.get(s):
                    translation_memory[s] = current[s]
                continue

            # 先尝试用翻译记忆直接复用，减少重复请求
            if s in translation_memory and translation_memory[s]:
                current[s] = translation_memory[s]
                reused_from_memory += 1
                continue

            source_locations[s].add(file_path)
            source_owner.setdefault(s, file_path)

    owner_to_strings: dict[str, dict[str, str]] = {}
    for source, owner in source_owner.items():
        owner_to_strings.setdefault(owner, {})[source] = ""

    if reused_from_memory:
        log.info("翻译记忆复用 %d 条", reused_from_memory)

    jobs: list[tuple[str, dict[str, str], str]] = []
    for file_path, to_translate in owner_to_strings.items():
        raw_content = _read_source_file(file_path, source_root)
        batches, file_content = split_batch(
            to_translate, system_prompt, file_path, raw_content,
        )
        for batch in batches:
            jobs.append((file_path, batch, file_content))

    tasks: list[asyncio.Task] = []
    for job_idx in range(len(jobs)):

        async def do_batch(
            idx: int = job_idx,
        ) -> tuple[int, dict[str, str], str | None]:
            fp, b, fc = jobs[idx]
            async with semaphore:
                translated, reason = await _translate_batch(
                    client, ai_cfg.model, fp, b, fc, system_prompt,
                )
                return idx, translated, reason

        tasks.append(asyncio.create_task(do_batch()))

    total = len(tasks)
    log.info("共 %d 个翻译批次，并发数 %d", total, ai_cfg.concurrency)
    pbar = ProgressBar(total, desc="翻译")

    failed_first_round: dict[int, str] = {}

    def apply_translations(translated_pairs: dict[str, str]) -> None:
        for source, translated in translated_pairs.items():
            if translated:
                translation_memory[source] = translated
            locations = source_locations.get(source)
            if locations:
                for fp in locations:
                    result.setdefault(fp, {})[source] = translated

    for coro in asyncio.as_completed(tasks):
        idx, translations, reason = await coro
        fp, batch, _ = jobs[idx]
        if translations:
            apply_translations(translations)
        else:
            failed_first_round[idx] = reason or "未知原因"
            log.warning(
                "批次失败 %d/%d: %s (%d 条) | %s",
                len(failed_first_round),
                total,
                fp,
                len(batch),
                failed_first_round[idx],
            )
        pbar.update(extra=f"失败 {len(failed_first_round)}")
    pbar.finish()

    # 对失败批次做一次低并发重试，降低限流影响
    if failed_first_round:
        retry_ids = list(failed_first_round.keys())
        log.info("开始重试失败批次: %d", len(retry_ids))
        retry_pbar = ProgressBar(len(retry_ids), desc="重试")
        recovered = 0
        for idx in retry_ids:
            fp, batch, file_content = jobs[idx]
            translated, reason = await _translate_batch(
                client, ai_cfg.model, fp, batch, file_content, system_prompt,
            )
            if translated:
                apply_translations(translated)
                recovered += 1
                del failed_first_round[idx]
            else:
                failed_first_round[idx] = (
                    f"{failed_first_round[idx]} | 重试后: {reason or '未知原因'}"
                )
            retry_pbar.update(extra=f"恢复 {recovered}")
        retry_pbar.finish()
        if recovered:
            log.info("重试恢复成功 %d 批次", recovered)

    failed_batches: list[dict[str, str | int]] = []
    for idx, reason in failed_first_round.items():
        fp, batch, _ = jobs[idx]
        affected_files = len({
            target_fp
            for source in batch
            for target_fp in source_locations.get(source, {fp})
        })
        failed_batches.append({
            "file": fp,
            "string_count": len(batch),
            "affected_files": affected_files,
            "reason": reason,
        })

    if failed_batches:
        log.warning("翻译阶段仍有失败批次: %d/%d", len(failed_batches), total)

    # AI 一致性修复（最多 2 轮）
    for fix_round in range(2):
        result, ai_fix_log = await _ai_fix_consistency(
            client, ai_cfg.model, system_prompt, result, glossary_path,
        )
        for msg in ai_fix_log:
            log.info("一致性修复 (第 %d 轮): %s", fix_round + 1, msg)
        if not ai_fix_log:
            break

    return result, failed_batches, total


def translate_all(
    strings_path: str,
    output_path: str,
    context_path: str = "",
    glossary_path: str = "config/glossary.yaml",
    mode: str = "incremental",
    lang: str = "zh-CN",
    ai_cfg: AIConfig | None = None,
    source_root: str = "",
) -> None:
    """同步入口"""
    if ai_cfg is None:
        ai_cfg = AIConfig()
    ai_cfg.validate()

    all_strings: TranslationDict = load_json(strings_path)
    existing = load_json(output_path) if Path(output_path).exists() else {}

    result, failed_batches, total_batches = asyncio.run(
        _translate_async(
            all_strings, existing, mode, lang,
            glossary_path, ai_cfg, source_root,
        )
    )
    # 全角 ASCII 符号统一转半角，避免破坏 Rust 源码语法
    for fp in result:
        for s, t in result[fp].items():
            if t:
                result[fp][s] = normalize_fullwidth(t)

    # 规则兜底：AI 修复后仍有问题的，用规则强制统一
    from .consistency import fix_consistency

    result, fix_log = fix_consistency(result, glossary_path)
    for msg in fix_log:
        log.info("规则兜底修复: %s", msg)

    save_json(result, output_path)
    log.info("翻译结果已保存: %s", output_path)

    if failed_batches:
        failure_log_path = f"{output_path}.failed_batches.json"
        by_file: dict[str, int] = {}
        for item in failed_batches:
            fp = str(item["file"])
            by_file[fp] = by_file.get(fp, 0) + 1
        file_summary = [
            {"file": fp, "failed_batches": n}
            for fp, n in sorted(by_file.items(), key=lambda x: x[1], reverse=True)
        ]
        save_json(
            {
                "total_batches": total_batches,
                "failed_batches": len(failed_batches),
                "failure_rate": round(
                    len(failed_batches) / max(total_batches, 1), 4,
                ),
                "by_file": file_summary,
                "items": failed_batches,
            },
            failure_log_path,
        )
        log.warning(
            "失败批次详情已保存: %s (%d/%d)",
            failure_log_path,
            len(failed_batches),
            total_batches,
        )


def run(args: argparse.Namespace) -> None:
    """CLI 入口"""
    ai_cfg = AIConfig(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        concurrency=args.concurrency,
    )
    translate_all(
        args.input,
        args.output,
        args.context,
        args.glossary,
        args.mode,
        args.lang,
        ai_cfg,
        source_root=getattr(args, "source_root", ""),
    )
