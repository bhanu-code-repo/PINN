"""MarkdownIndexer — orchestrates document parsing into trees.

Delegates parsing to ``markdown`` module and PDF conversion to ``pdf``
module. Handles summary generation via LLM (optional).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from loguru import logger

from ..models.schemas import DocumentTree, TreeNode
from . import markdown
from .pdf import pdf_to_markdown


class MarkdownIndexer:
    """Parse markdown (or PDF) into a hierarchical ``DocumentTree``.

    Parameters
    ----------
    min_node_tokens : int
        Tree-thinning threshold — nodes with fewer tokens (including
        descendants) are merged into their parent.
    summary_token_threshold : int
        Nodes with fewer tokens than this keep their full text as the
        summary instead of calling the LLM.
    enable_thinning : bool
        Whether to apply tree thinning.
    """

    def __init__(
        self,
        *,
        min_node_tokens: int = 50,
        summary_token_threshold: int = 200,
        enable_thinning: bool = True,
    ) -> None:
        self.min_node_tokens = min_node_tokens
        self.summary_token_threshold = summary_token_threshold
        self.enable_thinning = enable_thinning

    # -- Public API ---------------------------------------------------------

    async def index_file(
        self,
        path: str | Path,
        *,
        llm_client: object | None = None,
    ) -> DocumentTree:
        """Index a markdown or PDF file into a ``DocumentTree``.

        Parameters
        ----------
        path : str or Path
            Path to a ``.md`` or ``.pdf`` file.
        llm_client : LLMClient or None
            If provided, generates per-node summaries via LLM.
            If ``None``, summaries are left empty.
        """
        path = Path(path)
        content = self._read_content(path)

        # Parse markdown into flat node list
        node_list, lines = markdown.extract_headers(content)

        if not node_list:
            logger.debug("No headers found, creating single root node")
            node_list = [{"title": path.stem, "line_num": 1, "level": 1}]
            lines = content.split("\n")

        # Attach text content
        nodes_with_content = markdown.extract_text(node_list, lines)

        # Tree thinning
        if self.enable_thinning and len(nodes_with_content) > 1:
            nodes_with_content = markdown.compute_token_counts(nodes_with_content)
            nodes_with_content = markdown.thin_tree(nodes_with_content, self.min_node_tokens)

        # Build hierarchy with node IDs
        root_nodes = markdown.build_tree(nodes_with_content)

        # Generate summaries if LLM provided
        if llm_client is not None:
            await self._generate_summaries(root_nodes, llm_client)

        return DocumentTree(
            doc_name=path.stem,
            source_path=str(path),
            root_nodes=root_nodes,
        )

    def index_file_sync(
        self,
        path: str | Path,
        *,
        llm_client: object | None = None,
    ) -> DocumentTree:
        """Synchronous wrapper around :meth:`index_file`."""
        return asyncio.run(self.index_file(path, llm_client=llm_client))

    # -- Content reading ----------------------------------------------------

    @staticmethod
    def _read_content(path: Path) -> str:
        """Read file content, converting PDF to markdown if needed."""
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            return pdf_to_markdown(path)
        elif suffix in (".md", ".markdown", ".txt"):
            return path.read_text(encoding="utf-8")
        else:
            msg = f"Unsupported file format: {suffix}. Use .md, .txt, or .pdf"
            raise ValueError(msg)

    # -- Summary generation -------------------------------------------------

    async def _generate_summaries(
        self,
        nodes: list[TreeNode],
        llm_client: object,
    ) -> None:
        """Generate LLM summaries for all nodes in the tree."""
        flat = markdown.flatten_nodes(nodes)
        sem = asyncio.Semaphore(5)

        async def _summarize(node: TreeNode) -> None:
            token_count = markdown.count_tokens(node.text)
            if token_count < self.summary_token_threshold:
                node.summary = node.text
                return

            prompt = (
                f"Summarize the following section titled '{node.title}' "
                f"in 1-2 sentences. Focus on key findings, methods, or "
                f"conclusions.\n\n{node.text}"
            )

            async with sem:
                try:
                    node.summary = await llm_client.ask_async(prompt)  # type: ignore[union-attr]
                except Exception as exc:
                    logger.warning("Summary generation failed for {}: {}", node.node_id, exc)
                    node.summary = node.text[:200] + "..." if len(node.text) > 200 else node.text

        await asyncio.gather(*[_summarize(n) for n in flat])
