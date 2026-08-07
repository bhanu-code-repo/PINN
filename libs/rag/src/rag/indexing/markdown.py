"""Markdown parsing — extract headers, text, build tree structure.

Pure functions with no LLM dependency. Adapted from the vectorless-rag
PageIndex engine's ``page_index_md.py``.
"""

from __future__ import annotations

import re

from ..models.schemas import TreeNode

try:
    import litellm
except ImportError:  # pragma: no cover
    litellm = None  # type: ignore[assignment]


def extract_headers(content: str) -> tuple[list[dict], list[str]]:
    """Parse markdown headers, skipping those inside code blocks.

    Returns (node_list, lines) where each node has title, line_num, level.
    """
    header_pattern = re.compile(r"^(#{1,6})\s+(.+)$")
    code_block_pattern = re.compile(r"^```")
    node_list: list[dict] = []
    lines = content.split("\n")
    in_code_block = False

    for line_num, line in enumerate(lines, 1):
        stripped = line.strip()

        if code_block_pattern.match(stripped):
            in_code_block = not in_code_block
            continue

        if not in_code_block:
            match = header_pattern.match(stripped)
            if match:
                node_list.append({
                    "title": match.group(2).strip(),
                    "line_num": line_num,
                    "level": len(match.group(1)),
                })

    return node_list, lines


def extract_text(node_list: list[dict], lines: list[str]) -> list[dict]:
    """Attach text content between each header and the next."""
    result: list[dict] = []
    for i, node in enumerate(node_list):
        start = node["line_num"] - 1
        end = node_list[i + 1]["line_num"] - 1 if i + 1 < len(node_list) else len(lines)
        text = "\n".join(lines[start:end]).strip()
        result.append({**node, "text": text})
    return result


def count_tokens(text: str) -> int:
    """Approximate token count using litellm or word-based fallback."""
    if litellm is not None:
        try:
            return litellm.token_counter(model="gpt-4", text=text)
        except Exception:
            pass
    return len(text.split())


def compute_token_counts(node_list: list[dict]) -> list[dict]:
    """Add ``text_token_count`` including descendant text (bottom-up)."""
    result = [dict(n) for n in node_list]

    for i in range(len(result) - 1, -1, -1):
        level = result[i]["level"]
        total_text = result[i].get("text", "")

        for j in range(i + 1, len(result)):
            if result[j]["level"] <= level:
                break
            child_text = result[j].get("text", "")
            if child_text:
                total_text += "\n" + child_text

        result[i]["text_token_count"] = count_tokens(total_text)

    return result


def thin_tree(node_list: list[dict], min_node_tokens: int) -> list[dict]:
    """Merge nodes below ``min_node_tokens`` into their parent."""
    result = [dict(n) for n in node_list]
    to_remove: set[int] = set()

    for i in range(len(result) - 1, -1, -1):
        if i in to_remove:
            continue

        total_tokens = result[i].get("text_token_count", 0)
        if total_tokens >= min_node_tokens:
            continue

        level = result[i]["level"]
        children_texts: list[str] = []

        for j in range(i + 1, len(result)):
            if result[j]["level"] <= level:
                break
            if j not in to_remove:
                child_text = result[j].get("text", "")
                if child_text.strip():
                    children_texts.append(child_text)
                to_remove.add(j)

        if children_texts:
            merged = result[i].get("text", "")
            for ct in children_texts:
                if merged and not merged.endswith("\n"):
                    merged += "\n\n"
                merged += ct
            result[i]["text"] = merged
            result[i]["text_token_count"] = count_tokens(merged)

    for idx in sorted(to_remove, reverse=True):
        result.pop(idx)

    return result


def build_tree(node_list: list[dict]) -> list[TreeNode]:
    """Convert flat node list into nested ``TreeNode`` hierarchy."""
    if not node_list:
        return []

    stack: list[tuple[TreeNode, int]] = []
    root_nodes: list[TreeNode] = []
    counter = 1

    for node in node_list:
        level = node["level"]
        tree_node = TreeNode(
            node_id=str(counter).zfill(4),
            title=node["title"],
            text=node.get("text", ""),
            level=level,
        )
        counter += 1

        while stack and stack[-1][1] >= level:
            stack.pop()

        if not stack:
            root_nodes.append(tree_node)
        else:
            parent, _ = stack[-1]
            parent.children.append(tree_node)

        stack.append((tree_node, level))

    return root_nodes


def flatten_nodes(nodes: list[TreeNode]) -> list[TreeNode]:
    """Flatten tree into a list (pre-order traversal)."""
    result: list[TreeNode] = []
    for node in nodes:
        result.append(node)
        result.extend(flatten_nodes(node.children))
    return result
