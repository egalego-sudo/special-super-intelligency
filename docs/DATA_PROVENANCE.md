# Dataset Provenance, License Review and Training Gate

**Creator by Bittu Pandit**

The text dataset loader accepts JSON Lines records with mandatory non-empty fields:

- `text`: text content
- `source`: author, publisher, dataset or other source description
- `license`: exact license identifier or terms attached to that record

Example:

```json
{"text":"Example text","source":"Author-created sample","license":"CC0"}
```

## Mandatory training gate

Before text-based training, callers must provide an explicit `approved_licenses` allowlist. The training/evaluation pipeline rejects every record whose declared license is not on that exact allowlist. It produces a deterministic SHA-256 digest and per-source/per-license record counts for both training and validation splits.

This allowlist is an operator approval record, **not an automated legal opinion**. A human must inspect the original source, exact license version, attribution requirements, redistribution/training permissions, third-party content, and any contractual restrictions before adding a license identifier to the allowlist. A label such as `CC0` or `CC-BY-4.0` is not proof that the submitter had authority to license the content.

## Data handling

- The loader does not download data, scrape websites, or bypass access controls.
- Training and validation datasets are audited separately.
- Provenance metadata and dataset hashes should be preserved with every experiment.
- Do not use personal, confidential, copyrighted or restricted content unless its use is explicitly authorized.
- Synthetic fixtures in unit tests are only engineering checks; they do not establish real-world language capability.

The pipeline records provenance and enforces an explicit allowlist. It does not certify legal rights, data quality, representativeness, or benchmark validity.
