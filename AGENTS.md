# Contributor and agent guide

Read README.md and docs/RELEASE_STATUS.md before changing training code.

- Keep training seeds distinct from sampling seeds.
- Preserve recipe horizons when running short validation jobs.
- Verify global batch size and optimizer update counting after trainer changes.
- Run `python -m pytest -q` and `python scripts/check_release.py`.
- Validate distributed training and resume on CUDA before claiming GPU support.
- Keep credentials, machine-specific paths, datasets, weights and internal notes out.
- Preserve third-party attribution and document unverified functionality accurately.
