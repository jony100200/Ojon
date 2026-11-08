# Ojon — engineering rules

These rules apply equally to every contributor and coding agent. This is the **single shared project instruction file**; do not add provider-specific instruction documents.

## Product priorities

**1. Quantize models.** Make language, image, video and other supported AI models smaller and practical on local hardware, with measured quality. Improve the existing conversion, mixed-precision search, backend compatibility, and output validation. Several language-model and video workflows have already been executed; image-model quantization still requires demonstrated results.

**2. Uncensor models.** Make unwanted refusals editable through Ojon's existing weight-editing and automatic search workflows. Keep useful reasoning, coding and instruction-following abilities. Treat an edit as successful only when real before/after model runs demonstrate the intended change. Fix ineffective methods instead of claiming the entire product does not work.

Prioritize these two jobs. UI, packaging and utilities serve them; don't add unrelated frameworks, agent systems or speculative features.

## Execution rules

1. **Build and test before writing about building.** Inspect existing implementation, reuse working backends, make the smallest necessary code change, and run relevant checks.
2. **No unsolicited documents.** Do not create new Markdown files, policy pages, research reports, handoffs, status ledgers, roadmaps or redundant README files. Use existing guides only when a real command or feature changes, or when the owner explicitly requests documentation.
3. **Public presentation is for users.** Lead with what Ojon does and how to run it. No generic fear-based warnings, repeated disclaimers or blanket statements that the product is unusable. Describe specific supported formats and blockers precisely.
4. **Evidence over claims.** A source-code build, an edited model file, a runtime load, a successful quantization and a quality improvement are different results. Report what was actually measured; never fabricate success or dismiss verified work.
5. **Respect existing work.** One writer per checkout; use separate worktrees when needed. Never overwrite another agent's changes. Stage only intended paths, synchronize before pushing, and do not force-push.
6. **Respect running systems.** Check which models and workers are active before testing. Do not evict resident models, start persistent services, or modify other sessions' jobs without approval.
7. **Keep shared outputs portable.** No machine-specific paths, IP addresses, secrets, model weights or private datasets in public commits.
8. **Verification.** Run relevant Rust/Python tests and actual model runs when the task requires them. Say plainly when the hardware or checkpoint was not available. Use established test locations.
9. **Documentation gate.** Run `python scripts/check_public_docs.py` before public-doc changes. Do not expand its file allowlist without the owner's specific instruction.
10. **Finish with results.** Report changed code, exact tests performed, measured outcome and remaining engineering blocker. No ceremonial reports or extra documents.

Changes to public documentation without an explicit owner request require approval before pushing; ordinary tested code changes may go to `main` when the checkout is safe.
