"""
ingest.py — Offline RAG document ingestion pipeline
Raspberry Pi 5 Kiosk Project

Run this on your LAPTOP, then rsync the chromadb_index/ folder to the Pi.

Usage:
    python ingest.py                         # process all docs in ./docs/
    python ingest.py --input ./my_folder/    # custom input folder
    python ingest.py --input ./docs/ --watch # watch mode: re-index on new files
    python ingest.py --input ./docs/ --reset # wipe index and re-index everything

Requirements (install on your laptop):
    pip install docling chromadb sentence-transformers transformers watchdog
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import chromadb
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
# CONFIG — adjust these to match your setup
# ─────────────────────────────────────────────────────────────────────────────

DOCS_DIR        = Path("./docs")          # folder containing PDF / DOCX files
INDEX_DIR       = Path("./chromadb_index") # output: rsync this to Pi NVMe
COLLECTION_NAME = "kiosk_docs"
EMBED_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"
SUPPORTED_EXTS  = {".pdf", ".docx", ".pptx", ".html"}

CHUNK_MAX_TOKENS = 256    # target chunk size in tokens
CHUNK_OVERLAP    = 32     # token overlap between consecutive chunks

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
# STATE FILE — tracks which files have already been indexed
# Stored inside the index dir so it travels with the ChromaDB folder.
# ─────────────────────────────────────────────────────────────────────────────

STATE_FILE = INDEX_DIR / ".ingest_state.json"


def load_state() -> dict:
    """Returns {filepath: file_hash} for all previously indexed documents."""
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}


def save_state(state: dict):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def file_hash(path: Path) -> str:
    """MD5 hash of the file contents — used to detect changes."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# INGESTION CORE
# ─────────────────────────────────────────────────────────────────────────────

class Ingester:
    def __init__(self, reset: bool = False):
        log.info("Loading tokenizer: %s", EMBED_MODEL)
        self.tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)

        log.info("Initialising Docling converter")
        self.converter = DocumentConverter()

        self.chunker = HybridChunker(
            tokenizer=self.tokenizer,
            max_tokens=CHUNK_MAX_TOKENS,
            overlap=CHUNK_OVERLAP,
            merge_peers=True,   # merges short adjacent paragraphs
        )

        log.info("Opening ChromaDB at: %s", INDEX_DIR)
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        self.chroma = chromadb.PersistentClient(path=str(INDEX_DIR))

        if reset:
            log.warning("--reset flag set: deleting existing collection")
            try:
                self.chroma.delete_collection(COLLECTION_NAME)
            except Exception:
                pass

        # ChromaDB handles embedding internally when you pass documents as text.
        # We rely on its default embedding function (uses all-MiniLM-L6-v2
        # via chromadb's bundled sentence-transformers integration).
        self.collection = self.chroma.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},  # cosine similarity
        )

        self.state = load_state()

    # ── Per-document pipeline ─────────────────────────────────────────────────

    def ingest_file(self, path: Path) -> bool:
        """
        Parse → chunk → upsert one document.
        Returns True if the file was (re-)indexed, False if skipped.
        """
        path = path.resolve()
        key = str(path)
        current_hash = file_hash(path)

        # Skip unchanged files
        if self.state.get(key) == current_hash:
            log.info("SKIP (unchanged)  %s", path.name)
            return False

        log.info("INDEXING          %s", path.name)

        # ── 1. Parse with Docling ─────────────────────────────────────────────
        try:
            result = self.converter.convert(str(path))
        except Exception as e:
            log.error("Docling failed on %s: %s", path.name, e)
            return False

        doc = result.document

        # ── 2. Chunk with HybridChunker ───────────────────────────────────────
        chunks = list(self.chunker.chunk(doc))
        if not chunks:
            log.warning("No chunks produced for %s", path.name)
            return False

        log.info("  %d chunks produced", len(chunks))

        # ── 3. Remove old chunks for this file (handles re-indexing updates) ──
        self._delete_existing_chunks(path.name)

        # ── 4. Build IDs, texts, and metadata; upsert into ChromaDB ──────────
        ids, texts, metadatas = [], [], []

        for i, chunk in enumerate(chunks):
            text = chunk.text.strip()
            if not text:
                continue

            prov = None
            if chunk.meta.doc_items:
                first_item = chunk.meta.doc_items[0]
                if hasattr(first_item, "prov") and first_item.prov:
                    prov = first_item.prov[0]

            headings = " > ".join(chunk.meta.headings) if chunk.meta.headings else ""

            # Move content_hash up, before metadata
            content_hash = hashlib.md5(text.encode()).hexdigest()

            metadata = {
                "source":      path.name,
                "source_path": str(path),
                "page":        prov.page_no if prov else -1,
                "headings":    headings,
                "chunk_id":    content_hash,          # ← fixed: was chunk.meta.id
                "indexed_at":  datetime.utcnow().isoformat(),
            }

            doc_id = f"{path.stem}__{i}__{content_hash}"

            ids.append(doc_id)
            texts.append(text)
            metadatas.append(metadata)

        # ── Inspect first 3 chunks so you can verify quality ─────────────────
        self._preview_chunks(chunks[:3])

        # ── Upsert in batches of 100 (ChromaDB default limit) ────────────────
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            self.collection.upsert(
                ids=ids[i:i+batch_size],
                documents=texts[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size],
            )

        log.info("  Upserted %d chunks → ChromaDB", len(ids))

        # Mark file as indexed
        self.state[key] = current_hash
        save_state(self.state)
        return True

    def _delete_existing_chunks(self, filename: str):
        """Remove all previously indexed chunks from this file."""
        try:
            existing = self.collection.get(
                where={"source": filename},
                include=[],
            )
            if existing["ids"]:
                self.collection.delete(ids=existing["ids"])
                log.info("  Removed %d stale chunks for %s", len(existing["ids"]), filename)
        except Exception as e:
            log.warning("Could not remove old chunks for %s: %s", filename, e)

    def _preview_chunks(self, chunks):
        """Print a short preview of the first few chunks for quality checking."""
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

    # ── Batch processing ──────────────────────────────────────────────────────

    def ingest_folder(self, folder: Path):
        """Index all supported documents in a folder."""
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
                result = self.ingest_file(path)
                if result:
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
# WATCH MODE — re-index when files are added or modified
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

        # Debounce: editors write files multiple times on save
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

    # Initial index pass
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
# QUERY TEST — verify the index works before you rsync to Pi
# ─────────────────────────────────────────────────────────────────────────────

def test_query(ingester: Ingester, query: str):
    log.info("Test query: %r", query)
    results = ingester.collection.query(
        query_texts=[query],
        n_results=3,
        include=["documents", "metadatas", "distances"],
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    print("\n── Query results ──────────────────────────────────────")
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances)):
        score = 1 - dist   # cosine distance → similarity score
        print(f"\n[{i+1}] score={score:.3f}  source={meta['source']}  "
              f"page={meta['page']}  heading={meta['headings'][:50]}")
        print(f"    {doc[:200].strip()!r}")
    print("────────────────────────────────────────────────────────\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    global INDEX_DIR
    
    parser = argparse.ArgumentParser(
        description="Offline RAG ingestion pipeline — Docling + ChromaDB"
    )
    parser.add_argument(
        "--input", type=Path, default=DOCS_DIR,
        help=f"Folder containing PDF/DOCX files (default: {DOCS_DIR})"
    )
    parser.add_argument(
        "--index", type=Path, default=INDEX_DIR,
        help=f"ChromaDB output folder (default: {INDEX_DIR})"
    )
    parser.add_argument(
        "--watch", action="store_true",
        help="Watch input folder and re-index on changes"
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Wipe the existing index and re-index all documents"
    )
    parser.add_argument(
        "--test-query", type=str, metavar="QUERY",
        help="Run a test query against the built index and print top-3 results"
    )
    args = parser.parse_args()

    INDEX_DIR = args.index

    ingester = Ingester(reset=args.reset)

    if args.test_query:
        test_query(ingester, args.test_query)
        return

    if args.watch:
        watch_mode(ingester, args.input)
    else:
        ingester.ingest_folder(args.input)


if __name__ == "__main__":
    main()
