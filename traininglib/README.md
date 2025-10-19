# traininglib

Shared optimizer factories, scheduling utilities, and performance helpers used by the various model training pipelines in this repository. The package intentionally keeps its third-party dependencies tight (torch, transformers, and optional optimizer plugins) so specialised projects can reuse the training primitives without pulling the entire monorepo dependency set.
