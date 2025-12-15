<!-- Added to summarise remaining steps to fully enable the RAG LLM flow requested by users. -->
# RAG wiring status

- Historian Agent settings now surface the RAG toggle plus embedding, chunking, vector store, and hybrid weighting controls, and the API serialises/parses those fields so session overrides reach `HistorianAgentConfig`. <!-- Updated after wiring UI + backend to carry vector settings end-to-end. -->
- Operators still need to run `python docs/rag_llm_design/embed_existing_documents.py` before enabling vector retrieval; the script seeds the `document_chunks` collection and Chroma store but is not yet exposed via CLI or Flask. <!-- Keeping operational migration note so rollout steps remain clear. -->
- Ensure `CHROMA_PERSIST_DIRECTORY` (or the UI override) points at persistent storage and that `HISTORIAN_AGENT_USE_VECTOR_RETRIEVAL` is set to true once embeddings exist; otherwise the agent will fall back to keyword retrieval. <!-- Added reminder about persistence + fallback behaviour. -->
