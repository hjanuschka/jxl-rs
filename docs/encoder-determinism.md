# Encoder determinism notes

Current high-level encoder behavior is deterministic for repeated encodes of the same input
within the same build/configuration.

## Covered by tests

- Modular repeated same-input codestream equality.
- VarDCT repeated same-input codestream equality.
- Container output determinism with metadata boxes and `jxlp` chunking.
- Callback chunking path reassembles to exact byte-equal output.
- Representative high-level RGBA16/RGBA32f paths are deterministic.

## Scope

Determinism is currently guaranteed by effectively serial encode paths.
Cross-platform/threaded determinism is expected but not yet part of a formal hard guarantee.
