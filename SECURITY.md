# Security Policy

## Reporting a Vulnerability

Please email bugbounty@databricks.com to report any security vulnerabilities. We will acknowledge receipt of your vulnerability and strive to send you regular updates about our progress. If you're curious about the status of your disclosure please feel free to email us again. If you want to encrypt your disclosure email, you can use [this PGP key](https://keybase.io/arikfr/key.asc).

The same address covers anything else that should not be public: a credential,
personal or customer data, or third-party content without a compatible license
found in this repository. Please don't open a public issue for these.

## Owner response

The maintainers ([CODEOWNERS](CODEOWNERS)) act on a confirmed report promptly:
they remove or rotate the exposed material, fix the underlying cause, and rewrite
history where a secret or personal data was committed. Databricks may archive
the repository until a reported violation is resolved.

## Preventive controls

- **Secret scanning.** GitHub secret scanning and push protection on the
  repository, and pre-commit's `detect-private-key` on every commit
  (`.pre-commit-config.yaml`).
- **Dependency review.** Every direct dependency is pinned. The **Security**
  workflow (`.github/workflows/security.yml`) audits the Python and npm
  dependency trees against known vulnerabilities on every pull request, weekly,
  and before every deploy, and nothing merges or deploys while it fails.
  Dependabot proposes updates (`.github/dependabot.yml`).
- **Code scanning.** CodeQL analyzes the Python and JavaScript on every pull
  request.
- **No credentials in the repository.** Deployment settings live in each
  deployer's env file or in GitHub Environment variables, and CI authenticates
  with GitHub OIDC or an Environment secret scoped to the steps that need it.
- **Review.** Every change is reviewed before it merges ([CONTRIBUTING.md](CONTRIBUTING.md)).
