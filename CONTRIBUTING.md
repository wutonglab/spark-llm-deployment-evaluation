# Contributing

Thanks for your interest. This project provides a **reusable methodology** for evaluating LLM deployment decisions on DGX Spark and similar hardware. The most valuable contributions extend the methodology or add **case studies** that validate (or challenge) the frameworks on new model+hardware combinations.

## Ground Rules

1. **No proprietary, internal, or NDA-bound information.** All contributions must be based on publicly available information and reproducible measurements taken on hardware/software you have legitimate access to.
2. **Sanitize before submitting.** Hostnames, internal IPs, customer business details, employer-internal benchmarks — strip them before PR. CI runs `scripts/check-no-internal.sh` and will block merge on a hit.
3. **Show your work.** Framework scores must cite which channels (per frameworks 03/04) you consulted; benchmark numbers must include hardware, vLLM version, and the exact `variant-configs/*.env` used.

## Three Common Contribution Types

### A. Submit a New Case Study (most welcome)

Use [`case-studies/_template.md`](case-studies/_template.md):
1. Fork the repo, copy the template into `case-studies/<your-model>-on-<your-hardware>/`
2. Fill in framework scores with citations (channels you consulted)
3. Optionally include benchmark results (`data/<your>-results.csv`) and a `reproduce/` directory
4. Open a PR; we'll review for sanitization + framework consistency

### B. Improve a Framework

If you think a dimension's weight should change, a threshold is wrong, or a missing dimension matters:
1. Open an Issue using the "Framework feedback" template
2. Explain with evidence (cite case studies — yours or in the repo — that demonstrate the issue)
3. If accepted, PR the change to `frameworks/<N>-*.md` and update affected case studies' scores

### C. Improve the Agent / Tooling

Pull requests for `agent/`, `tools/`, `tests/` welcome. Please:
- Add a test in `tests/`
- Keep dependencies minimal (no LangChain/CrewAI etc. — we want a transparent loop)
- Don't change the LLM provider abstraction without discussion

## Pull Request Checklist

- [ ] Branch from `main`
- [ ] `bash scripts/check-no-internal.sh` passes (no sanitization hits)
- [ ] `pytest tests/` passes
- [ ] `ruff check .` clean
- [ ] If adding case study: framework scores cite ≥ 1 source per dimension
- [ ] If touching `frameworks/`: corresponding case studies re-scored to match
- [ ] If touching `agent/`: dry-run test (`pytest tests/test_agent_dryrun.py`) passes

## Discussion vs Issue

- **Discussions** for: sharing your case results (`reproduction-results` template), proposing framework changes (`framework-improvements` template), general Q&A
- **Issues** for: bugs, concrete feature requests, sanitization-hit reports

## Code of Conduct

This project follows the [Contributor Covenant 2.1](CODE_OF_CONDUCT.md).
