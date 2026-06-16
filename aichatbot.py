"""
aichatbot.py — AI Kiosk Chatbot
"""

import logging
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_ollama import OllamaLLM
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import ConversationalRetrievalChain
from pydantic import Field
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — must match ingest.py exactly
# ─────────────────────────────────────────────────────────────────────────────

INDEX_DIR       = Path("./chromadb_index")   # same path as ingest.py
COLLECTION_NAME = "kiosk_docs"               # same name as ingest.py
EMBED_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"  # same model as ingest.py
LLM_MODEL       = "llama3:8b"
K_RETRIEVAL     = 10

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("kiosk_chatbot")

# ─────────────────────────────────────────────────────────────────────────────
# CHROMADB RETRIEVER
# ─────────────────────────────────────────────────────────────────────────────

class ChromaDBRetriever(BaseRetriever):
    collection: Any = Field(...)
    k: int = Field(default=10)

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> list[Document]:
        results = self.collection.query(
            query_texts=[query],
            n_results=self.k,
            include=["documents", "metadatas", "distances"],
        )

        documents = []
        if results["documents"]:
            for doc_text, metadata in zip(
                results["documents"][0], results["metadatas"][0]
            ):
                source     = metadata.get("source", "unknown")
                page       = metadata.get("page", -1)
                headings   = metadata.get("headings", "")
                indexed_at = metadata.get("indexed_at", "")

                content = f"[{source}, page {page}] {doc_text}"
                if headings:
                    content = f"[{headings}]\n{content}"

                documents.append(Document(
                    page_content=content,
                    metadata={
                        "source":     source,
                        "page":       page,
                        "headings":   headings,
                        "indexed_at": indexed_at,
                    },
                ))

        return documents

# ─────────────────────────────────────────────────────────────────────────────
# INITIALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def init_chatbot():
    """Load the ChromaDB index built by ingest.py and wire up the LLM chain."""

    if not INDEX_DIR.exists():
        log.error("ChromaDB index not found at %s", INDEX_DIR)
        log.info("Run first:  python ingest.py")
        raise FileNotFoundError(f"Index directory '{INDEX_DIR}' does not exist")

    log.info("Loading ChromaDB from %s", INDEX_DIR)


    client = chromadb.PersistentClient(path=str(INDEX_DIR))

    try:
        collection = client.get_collection(
            name=COLLECTION_NAME,
        )
    except Exception as e:
        log.error("Collection '%s' not found: %s", COLLECTION_NAME, e)
        log.info("Run first:  python ingest.py")
        raise

    chunk_count = collection.count()
    log.info("Collection loaded — %d chunks indexed", chunk_count)

    if chunk_count == 0:
        log.warning("Collection is empty! Add documents and run ingest.py first.")

    retriever = ChromaDBRetriever(collection=collection, k=K_RETRIEVAL)
    llm       = OllamaLLM(model=LLM_MODEL)

    # ── Prompts ───────────────────────────────────────────────────────────────

    # Rewrites follow-up questions into self-contained queries
    condense_prompt = PromptTemplate(
        input_variables=["chat_history", "question"],
        template="""Given the conversation history and a follow-up question,
rephrase the follow-up into a clear, standalone question that can be
understood without the chat history.

Chat History:
{chat_history}

Follow-up question: {question}
Standalone question:""",
    )

    # Grounds the answer strictly in retrieved chunks
    qa_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a helpful kiosk assistant. Answer the visitor's question
using ONLY the information found in the context below.
If the answer is not covered by the context, respond with:
"I'm sorry, I don't have information about that in my knowledge base."
Do not make up, assume, or infer anything beyond what the context states.

Context:
{context}

Question: {question}

Answer:""",
    )

    # ── Chain ─────────────────────────────────────────────────────────────────
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        condense_question_prompt=condense_prompt,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        return_source_documents=True,
        verbose=False,
    )

    return qa_chain


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def print_sources(sources: list[Document]):
    """Pretty-print source documents from the last query."""
    if not sources:
        print("No sources available.\n")
        return

    print("\nSources:")
    seen = set()
    for doc in sources:
        key = (doc.metadata["source"], doc.metadata["page"])
        if key in seen:          # deduplicate same page cited multiple times
            continue
        seen.add(key)
        heading = doc.metadata.get("headings") or "—"
        print(f"  • {doc.metadata['source']}  "
              f"(page {doc.metadata['page']})  "
              f"| {heading}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CHAT LOOP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log.info("Initializing AI Kiosk Chatbot...")
    qa_chain = init_chatbot()

    print("\n" + "=" * 60)
    print("         AI Kiosk — Knowledge Base Assistant")
    print("=" * 60)
    print("  Ask any question answered by the loaded documents.")
    print("  Commands:")
    print("    sources  — show sources from the last answer")
    print("    clear    — reset conversation history")
    print("    exit / quit / q — exit")
    print("=" * 60 + "\n")

    chat_history: list[tuple[str, str]] = []
    last_sources: list[Document]        = []

    while True:
        try:
            query = input("You: ").strip()

            if not query:
                continue

            # ── Built-in commands ─────────────────────────────────────────────
            if query.lower() in ("exit", "quit", "q"):
                print("Thank you for using the kiosk. Goodbye!")
                break

            if query.lower() == "clear":
                chat_history.clear()
                last_sources.clear()
                print("Conversation history cleared.\n")
                continue

            if query.lower() == "sources":
                print_sources(last_sources)
                continue

            # ── RAG query ─────────────────────────────────────────────────────
            result = qa_chain.invoke({
                "question":    query,
                "chat_history": chat_history,
            })

            answer       = result["answer"]
            last_sources = result.get("source_documents", [])

            print(f"\nKiosk: {answer}\n")

            # Accumulate history for follow-up context
            chat_history.append((query, answer))

        except KeyboardInterrupt:
            print("\n\nSession ended. Goodbye!")
            break
        except Exception as e:
            log.error("Unexpected error: %s", e, exc_info=True)
            print("Something went wrong. Please try again.\n")


if __name__ == "__main__":
    main()