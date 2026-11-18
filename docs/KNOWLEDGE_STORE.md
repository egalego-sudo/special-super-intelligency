# Explicit Knowledge and Memory Interface

**Creator by Bittu Pandit**

The initial `KnowledgeStore` is an in-memory, provenance-aware index. It supports explicit record insertion, case-insensitive substring search, and exact-ID deletion.

## Privacy boundary

No user conversation is automatically captured. The caller must explicitly add a record. Each record retains a source label. This prototype is in-memory only and does not yet provide encryption, durable persistence, retention policies, multi-user isolation, or authenticated access controls.
