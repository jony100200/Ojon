# Ojon

Agent contract for this repository.

## Shared agent rules (all agents: Claude, GPT/Codex, AGY, Grok, humans)

The owner set these rules once. They apply in every Kalponic repository.

1. **No AI attribution anywhere.** No `Co-Authored-By:`, `Claude-Session:`, "Generated with …" or similar trailers in commits, PRs, issues or code comments. The owner is the author. Credit for outside work goes only in `ACKNOWLEDGMENTS.md`.
2. **Stage explicit paths.** Never `git add .`, `git add -A` or a whole folder. Never commit secrets, local config or build output.
3. **One writer per checkout.** Never overwrite or commit another agent's uncommitted work. On a shared checkout, work in your own worktree. Fetch and rebase right before every push. No force pushes.
4. **Direct commits to `main` are fine** when the checkout is safe and tests pass. Ask first only before destructive or hard-to-undo steps.
5. **No machine-local details in shared output.** No absolute paths, host names, IPs, ports, user names or keys in code, docs or tests. Load them from config, env or discovery.
6. **Never install boot, auto-start or scheduled services.** Give the owner the start command instead.
7. **Models: poke before inference.** Check that the exact model is loaded before calling it. Never silently load, unload or swap a model, or touch a model another session is using.
8. **Evidence over claims.** Never report a test, benchmark, model readiness or release that was not actually run. Report failures with their output.
9. **Tests live where the repo says.** Pothik's tests live in `Pothik-Tests`. Cross-project suites live in `MasterTests`.
10. **Plain English** in docs, commits and reports. Keep them short.
