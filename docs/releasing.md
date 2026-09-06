# PyPI release

The project `multimodal-recommender` has a published `0.1.0` release. New releases must use a
new version number; PyPI does not allow replacing files from an existing release.

## One-time account setup

1. Register at https://pypi.org/account/register/, verify your email and enable two-factor authentication.
2. Open https://pypi.org/manage/account/publishing/ and add a **pending publisher** for a new project:

   | Field | Value |
   | --- | --- |
   | PyPI project name | `multimodal-recommender` |
   | GitHub owner | `LazzyXP` |
   | Repository | `Multimodal-Recommender-System` |
   | Workflow filename | `release.yml` |
   | Environment | `pypi` |

3. In GitHub repository Settings → Environments, create `pypi`. Restrict release access as appropriate.

This uses short-lived GitHub OIDC credentials. No PyPI password or long-lived token belongs in the
repository or chat. Registering a pending publisher does not itself upload a package.

Official instructions: https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/

## Build, verify, then publish

1. Confirm `pyproject.toml` has the intended new version (currently the development version is `0.1.2`).
2. Wait for CI and **Package and publish** to pass on the exact commit being released.
   The packaging workflow builds both distributions, validates metadata with Twine, installs the
   source distribution, and tests the wheel on Linux/macOS/Windows with Python 3.11 and 3.12.
   The installed-package smoke test runs outside the checkout and exercises training, recommendation,
   save/load and the CLI. Optional torch models have their own existing CI job.
3. Create a GitHub Release with a matching tag, for example `v0.1.2`, targeting that tested commit.
   A plain push or manual workflow run only builds/tests; publishing requires a published Release.
4. The workflow verifies the tag against package metadata and uploads the tested artifacts to PyPI.
   Review the `publish` job result; a successful GitHub Release alone is not evidence of a PyPI upload.
5. Verify the matching PyPI version exists, then use a fresh environment:

   ```bash
   python -m venv /tmp/mmrec-pypi-check
   /tmp/mmrec-pypi-check/bin/python -m pip install --index-url https://pypi.org/simple multimodal-recommender==0.1.2
   /tmp/mmrec-pypi-check/bin/python scripts/package_smoke.py
   /tmp/mmrec-pypi-check/bin/mmrec --help
   ```

   The commands above are the **post-publication acceptance check**.
   On Windows use the virtual environment's `Scripts/python.exe` and `Scripts/mmrec.exe`.
6. Update the README's pinned installation version after the upload succeeds.

PyPI does not allow replacing an already uploaded version's files. Fixes after a successful upload
require a new version and matching tag. If only some files uploaded, inspect PyPI before retrying.
Do not create a release before account/publisher setup is complete.

## Local distribution verification

```bash
python -m pip install build twine
python -m build
python -m twine check --strict dist/*
python -m venv /tmp/mmrec-wheel-check
/tmp/mmrec-wheel-check/bin/python -m pip install dist/*.whl
/tmp/mmrec-wheel-check/bin/python scripts/package_smoke.py
/tmp/mmrec-wheel-check/bin/mmrec --help
```

Use a fresh output directory or remove only old distribution files before building, so stale versions
are not accidentally uploaded together. The GitHub workflow uses a clean checkout for every build.
