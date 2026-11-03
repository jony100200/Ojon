# Security policy

Ojon is under active pre-alpha development. Treat model files, downloaded backends and external tools as untrusted inputs, and keep original weights and data backed up.

## Report a vulnerability

Please **do not post secrets, private model data, access tokens, or exploit details in a public issue**.

If GitHub's private vulnerability reporting is enabled for this repository, use **Security → Report a vulnerability**. Otherwise contact the maintainer privately using a contact channel listed on their [GitHub profile](https://github.com/jony100200) before disclosing details publicly.

For normal bugs that do not expose sensitive information, use the [bug report template](https://github.com/jony100200/Ojon/issues/new/choose).

## Scope

Reports involving unintended file overwrites, path traversal, unsafe subprocess commands, secret leakage and misleading integrity verification are especially helpful.

Ojon does not claim that successful metadata checks establish a model's safety or output quality. Do not share confidential model weights or prompts with third-party services without reviewing the selected workflow.
