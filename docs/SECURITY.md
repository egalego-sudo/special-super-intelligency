# Security and Permission Boundaries

**Creator by Bittu Pandit**

## Current safeguards

- The initial tool registry invokes only explicitly registered in-process handlers.
- Unknown tool names are denied.
- Elevated and high-risk tools require an explicit human-approval flag.
- Tool arguments are type-checked.
- The registry does not provide shell execution, arbitrary code execution, network access, or secret access.

## Important limitation

An approval flag in this prototype is a local software gate, not an authenticated identity or secure approval workflow. A production tool runtime still needs OS/container isolation, least-privilege credentials, audit logs, timeouts, resource limits, and security review.

## Dependency hygiene

Dependabot is enabled for Python dependencies and GitHub Actions references. Updates must still be reviewed and tested before merging.
