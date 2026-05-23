# Security Policy

## Reporting a Vulnerability

If you find a security issue (e.g. credential leakage in committed files, supply-chain risk in the agent toolkit), please **do not** open a public GitHub issue.

Instead, open a GitHub Security Advisory via the "Security" tab → "Report a vulnerability".

## What This Project Does and Does Not Accept

### Accepted
- Reports of credentials, tokens, internal hostnames, IP addresses, or any non-public information accidentally committed to the repository
- Vulnerabilities in agent tool execution (shell injection, SSH key handling, arbitrary file write outside `evaluation-runs/`)
- Supply-chain issues in declared dependencies

### NOT Accepted
- Bug reports or feature requests containing **proprietary, customer, or NDA-bound information**
- Internal benchmark numbers from any organization
- Hardware roadmap leaks or unreleased product information

If your report contains such information, we cannot accept it — please redact before submitting.

## Disclosure Timeline

- Within 7 days: acknowledgement of receipt
- Within 30 days: triage + remediation plan
- Within 90 days: fix released (extended only by mutual agreement)
