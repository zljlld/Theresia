import argparse
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import requests


END_MARKER = "===END_INITIAL_PROMPT==="
MEMORY_SECTION_TITLE = "以下是之前对话的记忆"
MANAGER_PROMPT_TITLE = "# 记忆管理提示词"


@dataclass
class DeleteOp:
    created: str
    expire: str


@dataclass
class UpdateOp:
    created: str
    content: str
    expire: str


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def extract_manager_prompt(full_text: str) -> str:
    lines = full_text.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == MANAGER_PROMPT_TITLE:
            prompt_lines = []
            for j in range(i + 1, len(lines)):
                next_line = lines[j]
                if next_line.strip().startswith("# "):
                    break
                prompt_lines.append(next_line)
            return "\n".join(prompt_lines).strip()
    return ""


def split_initial_and_rest(full_text: str) -> tuple[str, str]:
    pos = full_text.find(END_MARKER)
    if pos == -1:
        raise ValueError(f"未找到标记: {END_MARKER}")
    return full_text[:pos], full_text[pos:]


def compose_full_text(initial_text: str, rest_text: str) -> str:
    """
    安全拼接 initial + rest，确保 END_MARKER 独占一行，避免粘连到记忆行末尾。
    """
    initial_clean = initial_text.rstrip("\n")
    rest_clean = rest_text.lstrip("\n")
    return initial_clean + "\n" + rest_clean


def collect_memory_lines(initial_text: str) -> list[str]:
    return [line for line in initial_text.splitlines() if line.strip().startswith("<")]


def build_management_question(manager_prompt: str, memory_lines: list[str]) -> str:
    memories = "\n".join(memory_lines) if memory_lines else "(空)"
    return f"{manager_prompt}\n\n当前记忆如下：\n{memories}"


def call_deepseek(api_key: str, question: str, model: str) -> str:
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": question}],
        "temperature": 0,
        "stream": False,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def parse_response_ops(response_text: str) -> list[DeleteOp | UpdateOp]:
    if response_text.strip().upper() == "NO":
        return []

    ops: list[DeleteOp | UpdateOp] = []
    lines = [line.strip() for line in response_text.splitlines() if line.strip()]

    delete_re = re.compile(
        r'^D\s*<(?P<created>\d{4}/\d{2}/\d{2}/\d{2}:\d{2})>\s*<(?P<expire>\d{4}/\d{2}/\d{2}/\d{2}:\d{2})>\s*$'
    )
    update_re = re.compile(
        r'^X\s*<(?P<created>\d{4}/\d{2}/\d{2}/\d{2}:\d{2})>(?P<content>.+)<(?P<expire>\d{4}/\d{2}/\d{2}/\d{2}:\d{2})>\s*$'
    )

    for line in lines:
        m_del = delete_re.match(line)
        if m_del:
            ops.append(DeleteOp(created=m_del.group("created"), expire=m_del.group("expire")))
            continue

        m_upd = update_re.match(line)
        if m_upd:
            ops.append(
                UpdateOp(
                    created=m_upd.group("created"),
                    content=m_upd.group("content").strip(),
                    expire=m_upd.group("expire"),
                )
            )
            continue

        raise ValueError(f"无法解析返回格式: {line}")

    return ops


def match_memory_line(line: str, created: str, expire: str) -> bool:
    s = line.strip()
    return s.startswith(f"<{created}>") and s.endswith(f"<{expire}>")


def parse_memory_line(line: str) -> tuple[str, str, str] | None:
    """
    Parse a memory line:
    <创建时间> 内容 <截至时间>
    """
    m = re.match(
        r'^\s*<(?P<created>\d{4}/\d{2}/\d{2}/\d{2}:\d{2})>\s*(?P<content>.+)\s*<(?P<expire>\d{4}/\d{2}/\d{2}/\d{2}:\d{2})>\s*$',
        line.strip(),
    )
    if not m:
        return None
    return m.group("created"), m.group("content").strip(), m.group("expire")


def normalize_memory_content(content: str) -> str:
    """
    Normalize content for dedupe:
    - unify Chinese punctuation to ASCII commas
    - collapse spaces
    - lower-case Latin letters
    """
    s = content.replace("，", ",").replace("：", ":").replace("。", ".")
    s = re.sub(r"\s+", " ", s).strip().lower()
    # Normalize spacing around punctuation to improve duplicate matching.
    s = re.sub(r"\s*,\s*", ",", s)
    s = re.sub(r"\s*:\s*", ":", s)
    return s


def dedupe_memory_lines(lines: list[str]) -> list[str]:
    """
    Remove near-duplicate memory lines by normalized content,
    keeping the latest one (last occurrence).
    """
    kept_reversed: list[str] = []
    seen: set[str] = set()

    for line in reversed(lines):
        parsed = parse_memory_line(line)
        if not parsed:
            kept_reversed.append(line)
            continue
        _, content, _ = parsed
        key = normalize_memory_content(content)
        if key in seen:
            continue
        seen.add(key)
        kept_reversed.append(line)

    return list(reversed(kept_reversed))


def _dup_group_key(line: str) -> str | None:
    """
    Key for duplicate-group protection:
    normalized content only
    """
    parsed = parse_memory_line(line)
    if not parsed:
        return None
    _, content, _ = parsed
    return normalize_memory_content(content)


def protect_duplicate_groups(original_lines: list[str], current_lines: list[str]) -> tuple[list[str], int]:
    """
    Protection rule:
    if a duplicate group (count > 1 in original) is fully deleted after ops,
    restore one latest line from that group.
    """
    original_groups: dict[str, list[str]] = {}
    for line in original_lines:
        key = _dup_group_key(line)
        if key is None:
            continue
        original_groups.setdefault(key, []).append(line)

    # Only groups that were duplicates in original input.
    dup_keys = {k for k, v in original_groups.items() if len(v) > 1}
    if not dup_keys:
        return current_lines, 0

    current_group_counts: dict[str, int] = {}
    for line in current_lines:
        key = _dup_group_key(line)
        if key is None:
            continue
        current_group_counts[key] = current_group_counts.get(key, 0) + 1

    restored = 0
    new_lines = list(current_lines)
    for key in dup_keys:
        if current_group_counts.get(key, 0) == 0:
            # Restore the latest one from original duplicate group (last occurrence).
            keep_line = original_groups[key][-1]
            new_lines.append(keep_line)
            restored += 1

    return new_lines, restored


def reorder_memory_lines(lines: list[str]) -> list[str]:
    """
    Reorder parsed memory lines by:
    1) longer duration first (expire - created, desc)
    2) if same duration, older created time first (asc)
    Non-memory/malformed lines keep their original positions.
    """
    memory_positions: list[int] = []
    sortable_items: list[tuple[float, datetime, str]] = []

    for idx, line in enumerate(lines):
        parsed = parse_memory_line(line)
        if not parsed:
            continue
        created, _, expire = parsed
        try:
            created_dt = datetime.strptime(created, "%Y/%m/%d/%H:%M")
            expire_dt = datetime.strptime(expire, "%Y/%m/%d/%H:%M")
            duration_seconds = (expire_dt - created_dt).total_seconds()
        except ValueError:
            # If parsing fails, skip sorting for this line.
            continue

        memory_positions.append(idx)
        sortable_items.append((duration_seconds, created_dt, line))

    # duration desc, created asc
    sortable_items.sort(key=lambda x: (-x[0], x[1]))

    reordered = list(lines)
    for pos, item in zip(memory_positions, sortable_items):
        reordered[pos] = item[2]

    return reordered


def apply_ops_to_initial(initial_text: str, ops: list[DeleteOp | UpdateOp]) -> tuple[str, int]:
    original_lines = initial_text.splitlines()
    lines = list(original_lines)
    changed = 0

    for op in ops:
        if isinstance(op, DeleteOp):
            # Delete all lines matching created+expire.
            before_count = len(lines)
            lines = [line for line in lines if not match_memory_line(line, op.created, op.expire)]
            changed += (before_count - len(lines))

        if isinstance(op, UpdateOp):
            replaced = False
            for idx, line in enumerate(lines):
                if match_memory_line(line, op.created, op.expire):
                    lines[idx] = f"<{op.created}> {op.content} <{op.expire}>"
                    replaced = True
                    changed += 1
                    break
            if not replaced:
                # 找不到则补一条，避免模型给了更新指令但无法生效
                lines.append(f"<{op.created}> {op.content} <{op.expire}>")
                changed += 1

    # Safety net: post-dedupe memory lines even if model misses some duplicates.
    deduped = dedupe_memory_lines(lines)
    changed += max(0, len(lines) - len(deduped))
    lines = deduped

    # Safety rule: avoid deleting an entire duplicate group accidentally.
    lines, restored_count = protect_duplicate_groups(original_lines, lines)
    changed += restored_count

    # Apply deterministic ordering to memory lines.
    reordered = reorder_memory_lines(lines)
    if reordered != lines:
        changed += 1
        lines = reordered

    return "\n".join(lines), changed


def manage_memories(prompt_file: Path, api_key: str, model: str, dry_run: bool) -> dict:
    full_text = read_text(prompt_file)
    manager_prompt = extract_manager_prompt(full_text)
    if not manager_prompt:
        raise ValueError("未找到记忆管理提示词内容")

    initial, rest = split_initial_and_rest(full_text)
    memory_lines = collect_memory_lines(initial)

    question = build_management_question(manager_prompt, memory_lines)
    answer = call_deepseek(api_key, question, model)
    ops = parse_response_ops(answer)

    # Always run safety normalize passes (dedupe + reorder) on current memories.
    original_lines = initial.splitlines()
    deduped_lines = dedupe_memory_lines(original_lines)
    reordered_lines = reorder_memory_lines(deduped_lines)
    normalized_initial = "\n".join(reordered_lines)
    normalize_changed = normalized_initial != initial

    if not ops:
        if normalize_changed:
            if not dry_run:
                write_text(prompt_file, compose_full_text(normalized_initial, rest))
            changed_count = 0
            changed_count += max(0, len(original_lines) - len(deduped_lines))
            if reordered_lines != deduped_lines:
                changed_count += 1
            return {
                "status": "updated",
                "answer": "NO",
                "changed_count": max(1, changed_count),
                "written": not dry_run,
            }
        return {
            "status": "no_change",
            "answer": answer,
            "changed_count": 0,
            "written": False,
        }

    updated_initial, changed_count = apply_ops_to_initial(initial, ops)

    if dry_run:
        return {
            "status": "updated",
            "answer": answer,
            "changed_count": changed_count,
            "written": False,
        }

    write_text(prompt_file, compose_full_text(updated_initial, rest))
    return {
        "status": "updated",
        "answer": answer,
        "changed_count": changed_count,
        "written": True,
    }


def run_manager(prompt_file: Path, api_key: str, model: str, dry_run: bool) -> None:
    result = manage_memories(prompt_file, api_key, model, dry_run)
    if result["status"] == "no_change":
        print("记忆管理结果: NO（无需修改）")
        return

    print(f"管理指令: {result['answer']}")
    print(f"变更条数: {result['changed_count']}")
    if dry_run:
        print("dry-run 模式，未写回文件。")
    elif result["written"]:
        print(f"已写回: {prompt_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="根据“记忆管理提示词”调用 DeepSeek 管理 system_prompt 记忆。")
    parser.add_argument("--prompt-file", default="system_prompt.txt", help="提示词文件路径，默认 system_prompt.txt")
    parser.add_argument("--api-key", default=os.getenv("DEEPSEEK_API_KEY", ""), help="DeepSeek API Key，可用环境变量 DEEPSEEK_API_KEY")
    parser.add_argument("--model", default="deepseek-chat", help="模型名，默认 deepseek-chat")
    parser.add_argument("--dry-run", action="store_true", help="仅打印结果，不写回文件")
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("缺少 API Key。请通过 --api-key 或 DEEPSEEK_API_KEY 提供。")

    run_manager(Path(args.prompt_file), args.api_key, args.model, args.dry_run)


if __name__ == "__main__":
    main()
