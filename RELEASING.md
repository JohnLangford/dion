# Releasing Dion

Dion publishes to [PyPI](https://pypi.org/p/dion) from
`.github/workflows/release.yml` when a `v*` tag is pushed. Publishing uses PyPI
[Trusted Publishing](https://docs.pypi.org/trusted-publishers/), so no API token
is stored in this repository.

## One-time setup

Both steps need a human with the right accounts; neither can be done from a PR.

**1. Register the pending publisher on PyPI.** The `dion` name is unclaimed, so
the first upload creates the project. Log in to pypi.org with the account that
should own it, go to *Your projects → Publishing → Add a new pending publisher*,
and enter:

| Field | Value |
| --- | --- |
| PyPI Project Name | `dion` |
| Owner | `microsoft` |
| Repository name | `dion` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

A pending publisher is what lets the workflow create a project that does not
exist yet. Once the first release lands it becomes an ordinary trusted publisher.

**2. Create the `pypi` environment.** In *Settings → Environments*, add an
environment named `pypi`. The name must match the workflow and the pending
publisher exactly. Adding required reviewers here puts a human approval in front
of every upload, which is worth doing on a public package.

## Cutting a release

1. Bump `version` in `setup.py`. Versions below `1.0` are pre-release: breaking
   changes are allowed, but a released version number can never be reused.
2. Move the `[Unreleased]` entries in `CHANGELOG.md` under a new
   `## [X.Y.Z] - YYYY-MM-DD` heading.
3. Open a PR with both changes and merge it. The `Release` workflow builds and
   validates the distributions on that PR, so packaging breakage surfaces before
   the tag exists.
4. Tag the merge commit and push:

   ```bash
   git tag vX.Y.Z && git push origin vX.Y.Z
   ```

5. The workflow rebuilds, verifies the tag matches `setup.py`'s version, and
   publishes. If the `pypi` environment requires reviewers, approve the run.

## Notes

- **A version is permanent.** PyPI does not allow reuse of a version number or
  of a distribution filename, even after deletion. A bad release is yanked and
  superseded, never replaced. Test on TestPyPI first if a release is unusual:
  configure a second pending publisher at test.pypi.org and run the workflow
  against it, or upload once by hand with
  `twine upload -r testpypi dist/*`.
- **`twine check` is not a metadata check.** It renders the long description and
  little else. The fields PyPI actually validates on upload — `author_email`
  above all — are covered by `tests/test_packaging.py`, which runs in CI.
- **The sdist has to carry `requirements_*.txt`.** `setup.py` reads them at build
  time to populate `install_requires`; if `MANIFEST.in` stops shipping them, a
  build from the sdist still succeeds but declares no dependencies at all.
  `tests/test_packaging.py` asserts against the built artifacts to catch this.
