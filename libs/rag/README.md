# rag — Structure-Based Retrieval-Augmented Generation

A vectorless RAG library that uses document structure as the index instead of
vector embeddings. Parses markdown and PDF documents into hierarchical trees,
indexes with BM25 keyword search, and retrieves relevant sections via LLM
reasoning over summaries.

## Quick Start

```python
from rag import MarkdownIndexer, KnowledgeStore, retrieve

# Index a document (PDF or Markdown)
indexer = MarkdownIndexer()
tree = indexer.index_file_sync("paper.pdf")      # auto-detects format
tree = indexer.index_file_sync("notes.md")        # same API

# Store it
store = KnowledgeStore("./my-knowledge-base")
store.add_document(tree)
store.save()

# Retrieve relevant sections
from llm_provider import LLMClient
results = await retrieve(store, "How to handle shock waves in PINNs?", LLMClient())
```

## How It Works

1. **Index**: Parse document headers into a tree, extract text per section,
   optionally generate LLM summaries per node
2. **Store**: Save indexed trees as JSON with metadata (keywords, techniques,
   PDE types)
3. **Search**: BM25 keyword pre-filter narrows candidates from the full corpus
4. **Retrieve**: LLM reasons over summarized trees (titles + summaries) to
   select relevant nodes, then fetches full text for those nodes only

No vector database, no embeddings, no chunking artifacts. The document
structure *is* the index.

## PDF Support

PDFs are automatically converted to markdown via `pymupdf4llm` (zero LLM cost,
font-size-based header detection), then indexed through the same markdown
pipeline. One API for both formats.
