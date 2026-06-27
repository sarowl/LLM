"""
ingest.py — Offline RAG document ingestion pipeline
Raspberry Pi 5 Kiosk Project

Run this on your LAPTOP, then rsync the chromadb_index/ folder to the Pi.

Usage:
    python ingest.py                         # process all docs in ./docs/
    python ingest.py --input ./my_folder/    # custom input folder
    python ingest.py --input ./docs/ --watch # watch mode: re-index on new files
    python ingest.py --input ./docs/ --reset # wipe index and re-index everything
    python ingest.py --test-query "your question here"

Requirements (install on your laptop):
    pip install docling chromadb sentence-transformers transformers watchdog rank_bm25 python-dotenv
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from transformers import AutoTokenizer

# ── Watchdog is only needed for --watch mode ──────────────────────────────────
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — all values can be overridden via .env file
# ─────────────────────────────────────────────────────────────────────────────

DOCS_DIR        = Path(os.getenv("DOCS_DIR",        "./docs"))
INDEX_DIR       = Path(os.getenv("INDEX_DIR",       "./chromadb_index"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME",      "kiosk_docs")
EMBED_MODEL     = os.getenv("EMBED_MODEL",          "sentence-transformers/all-MiniLM-L6-v2")
SUPPORTED_EXTS  = {".pdf", ".docx", ".pptx", ".html"}

CHUNK_MAX_TOKENS = int(os.getenv("CHUNK_MAX_TOKENS", "512"))
CHUNK_OVERLAP    = int(os.getenv("CHUNK_OVERLAP",    "64"))

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ingest")

# ─────────────────────────────────────────────────────────────────────────────
# STATE FILE
# ─────────────────────────────────────────────────────────────────────────────

def _state_file(index_dir: Path) -> Path:
    return index_dir / ".ingest_state.json"


def load_state(index_dir: Path) -> dict:
    sf = _state_file(index_dir)
    if sf.exists():
        with open(sf) as f:
            return json.load(f)
    return {}


def save_state(state: dict, index_dir: Path):
    sf = _state_file(index_dir)
    sf.parent.mkdir(parents=True, exist_ok=True)
    with open(sf, "w") as f:
        json.dump(state, f, indent=2)


def file_hash(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65_536), b""):
            h.update(block)
    return h.hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# INGESTER
# ─────────────────────────────────────────────────────────────────────────────

class Ingester:
    """
    Parse → chunk → embed → upsert pipeline.
    index_dir is a constructor parameter (no global mutation).
    Embedding function is explicit so ingest and retrieval always match.
    """

    def __init__(self, index_dir: Path = INDEX_DIR, reset: bool = False):
        self.index_dir = index_dir

        log.info("Loading tokenizer: %s", EMBED_MODEL)
        self.tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)

        self.embed_fn = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)

        log.info("Initialising Docling converter")
        self.converter = DocumentConverter()

        self.chunker = HybridChunker(
            tokenizer=self.tokenizer,
            max_tokens=CHUNK_MAX_TOKENS,
            overlap=CHUNK_OVERLAP,
            merge_peers=True,
        )

        log.info("Opening ChromaDB at: %s", self.index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.chroma = chromadb.PersistentClient(path=str(self.index_dir))

        if reset:
            log.warning("--reset flag set: deleting existing collection")
            try:
                self.chroma.delete_collection(COLLECTION_NAME)
            except Exception:
                pass

        self.collection = self.chroma.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

        self.state = load_state(self.index_dir)

    # ── Per-document pipeline ─────────────────────────────────────────────────

    def ingest_file(self, path: Path) -> bool:
        """Parse → chunk → upsert one document. Returns True if (re-)indexed."""
        path = path.resolve()
        key = str(path)
        current_hash = file_hash(path)

        if self.state.get(key) == current_hash:
            log.info("SKIP (unchanged)  %s", path.name)
            return False

        log.info("INDEXING          %s", path.name)

        try:
            result = self.converter.convert(str(path))
        except Exception as e:
            log.error("Docling failed on %s: %s", path.name, e)
            return False

        doc = result.document
        chunks = list(self.chunker.chunk(doc))
        if not chunks:
            log.warning("No chunks produced for %s", path.name)
            return False

        log.info("  %d chunks produced", len(chunks))
        self._delete_existing_chunks(path.name)

        ids, texts, metadatas = [], [], []
        now_iso = datetime.now(timezone.utc).isoformat()

        for i, chunk in enumerate(chunks):
            text = chunk.text.strip()
            if not text:
                continue

            prov = None
            if chunk.meta.doc_items:
                first_item = chunk.meta.doc_items[0]
                if hasattr(first_item, "prov") and first_item.prov:
                    prov = first_item.prov[0]

            headings_list = chunk.meta.headings or []
            headings_path = " > ".join(headings_list)
            top_heading   = headings_list[0] if headings_list else ""

            content_hash = hashlib.md5(text.encode()).hexdigest()

            metadata = {
                "source":      path.name,
                "source_path": str(path),
                "page":        prov.page_no if prov else -1,
                "headings":    headings_path,
                "top_heading": top_heading,
                "chunk_id":    content_hash,
                "indexed_at":  now_iso,
            }

            doc_id = f"{path.stem}__{i}__{content_hash}"
            ids.append(doc_id)
            texts.append(text)
            metadatas.append(metadata)

        self._preview_chunks(chunks[:3])

        batch_size = 100
        for i in range(0, len(ids), batch_size):
            self.collection.upsert(
                ids=ids[i : i + batch_size],
                documents=texts[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size],
            )

        log.info("  Upserted %d chunks → ChromaDB", len(ids))
        self.state[key] = current_hash
        save_state(self.state, self.index_dir)
        return True

    def _delete_existing_chunks(self, filename: str):
        try:
            existing = self.collection.get(where={"source": filename}, include=[])
            if existing["ids"]:
                self.collection.delete(ids=existing["ids"])
                log.info("  Removed %d stale chunks for %s", len(existing["ids"]), filename)
        except Exception as e:
            log.warning("Could not remove old chunks for %s: %s", filename, e)

    def _preview_chunks(self, chunks):
        log.info("  ── Chunk preview ──────────────────────────")
        for i, chunk in enumerate(chunks):
            tokens = self.tokenizer.encode(chunk.text)
            prov = None
            if chunk.meta.doc_items and chunk.meta.doc_items[0].prov:
                prov = chunk.meta.doc_items[0].prov[0]
            log.info(
                "  [%02d] tokens=%-4d page=%-3s heading=%s",
                i,
                len(tokens),
                prov.page_no if prov else "?",
                (chunk.meta.headings or ["(none)"])[0][:40],
            )
            log.info("       %r", chunk.text[:100].strip())
        log.info("  ────────────────────────────────────────────")

    # ── Batch ─────────────────────────────────────────────────────────────────

    def ingest_folder(self, folder: Path):
        files = [
            f for f in folder.rglob("*")
            if f.suffix.lower() in SUPPORTED_EXTS and f.is_file()
        ]
        if not files:
            log.warning("No supported files found in %s", folder)
            log.warning("Supported: %s", ", ".join(SUPPORTED_EXTS))
            return

        log.info("Found %d file(s) in %s", len(files), folder)
        indexed, skipped, failed = 0, 0, 0

        for path in files:
            try:
                # FIX: original called ingest_file twice per path,
                # doubling work and miscounting stats
                did_index = self.ingest_file(path)
                if did_index:
                    indexed += 1
                else:
                    skipped += 1
            except Exception as e:
                log.error("Unexpected error on %s: %s", path.name, e)
                failed += 1

        log.info("")
        log.info("Done — indexed: %d  skipped: %d  failed: %d", indexed, skipped, failed)
        log.info("Total chunks in collection: %d", self.collection.count())
        log.info("")
        log.info("Next step — copy index to Pi:")
        log.info("  rsync -avz ./chromadb_index/ pi@raspberrypi.local:/mnt/nvme/chromadb/")


# ─────────────────────────────────────────────────────────────────────────────
# WATCH MODE
# ─────────────────────────────────────────────────────────────────────────────

class DocWatcher(FileSystemEventHandler):
    def __init__(self, ingester: Ingester, folder: Path):
        self.ingester = ingester
        self.folder = folder
        self._debounce: dict[str, float] = {}

    def on_modified(self, event):
        self._handle(event.src_path)

    def on_created(self, event):
        self._handle(event.src_path)

    def _handle(self, src_path: str):
        path = Path(src_path)
        if path.suffix.lower() not in SUPPORTED_EXTS:
            return
        now = time.time()
        if now - self._debounce.get(src_path, 0) < 2.0:
            return
        self._debounce[src_path] = now
        log.info("Detected change: %s", path.name)
        self.ingester.ingest_file(path)


def watch_mode(ingester: Ingester, folder: Path):
    if not WATCHDOG_AVAILABLE:
        log.error("watchdog not installed. Run: pip install watchdog")
        sys.exit(1)
    ingester.ingest_folder(folder)
    log.info("Watching %s for changes — Ctrl+C to stop", folder)
    handler = DocWatcher(ingester, folder)
    observer = Observer()
    observer.schedule(handler, str(folder), recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


# ─────────────────────────────────────────────────────────────────────────────
# TEST QUERY
# ─────────────────────────────────────────────────────────────────────────────

def test_query(ingester: Ingester, query: str):
    log.info("Test query: %r", query)
    results = ingester.collection.query(
        query_texts=[query],
        n_results=3,
        include=["documents", "metadatas", "distances"],
    )
    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]

    print("\n── Query results ──────────────────────────────────────")
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances)):
        score = 1 - dist
        print(f"\n[{i+1}] score={score:.3f}  source={meta['source']}  "
              f"page={meta['page']}  heading={meta['headings'][:50]}")
        print(f"    {doc[:200].strip()!r}")
    print("────────────────────────────────────────────────────────\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Offline RAG ingestion pipeline — Docling + ChromaDB"
    )
    parser.add_argument("--input",  type=Path, default=DOCS_DIR,
                        help=f"Folder containing PDF/DOCX files (default: {DOCS_DIR})")
    parser.add_argument("--index",  type=Path, default=INDEX_DIR,
                        help=f"ChromaDB output folder (default: {INDEX_DIR})")
    parser.add_argument("--watch",  action="store_true",
                        help="Watch input folder and re-index on changes")
    parser.add_argument("--reset",  action="store_true",
                        help="Wipe the existing index and re-index all documents")
    parser.add_argument("--test-query", type=str, metavar="QUERY",
                        help="Run a test query and print top-3 results")
    args = parser.parse_args()

    ingester = Ingester(index_dir=args.index, reset=args.reset)

    if args.test_query:
        test_query(ingester, args.test_query)
        return

    if args.watch:
        watch_mode(ingester, args.input)
    else:
        ingester.ingest_folder(args.input)


if __name__ == "__main__":
    main()