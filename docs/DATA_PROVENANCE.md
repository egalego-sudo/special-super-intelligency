# Dataset Provenance and Loading

**Creator by Bittu Pandit**

The initial dataset loader accepts JSON Lines records with mandatory non-empty fields:

- `text`: text content
- `source`: provenance/source description
- `license`: declared license identifier or terms

Records without provenance or a license declaration are rejected rather than silently accepted. This is a metadata gate, not a legal determination that a dataset is safe to use. Human review of source terms and permitted use is still required.

Example:

```json
{"text":"Example text","source":"Author-created sample","license":"CC0"}
```

The loader does not download data, scrape websites, or bypass access controls.
