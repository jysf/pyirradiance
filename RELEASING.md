# Releasing

Releases go to [PyPI](https://pypi.org/project/irradiance/) as `irradiance`
(wheels only) and to [GitHub releases](https://github.com/jysf/pyirradiance/releases).
CI does the publishing; nothing is uploaded by hand.

## Steps

1. Get `main` to the state you want to ship, pushed, with CI green.
2. Run:

   ```sh
   just release 0.0.3
   ```

   It first checks (`just release-check 0.0.3` runs the checks alone):
   - the version is `X.Y.Z`, on `main`, clean tree, in sync with `origin/main`;
   - tag `v0.0.3` doesn't exist and PyPI doesn't have `0.0.3`;
   - every CI run on `HEAD` succeeded.

   Then it bumps `version` in `pyproject.toml` and `Cargo.toml` (and
   `Cargo.lock`), commits "Release v0.0.3", tags `v0.0.3` and pushes both.
3. The tag starts `.github/workflows/wheels.yml`:
   1. **build**: one abi3 wheel per platform (Linux x86_64 and aarch64 as
      manylinux2014, macOS universal2). Each is installed into a clean venv and
      runs the full suite (`tests/ci.sh`) on Python 3.12, plus the golden
      hashes on 3.9.
   2. **pypi**: checks the tag matches `pyproject.toml`'s version, then uploads
      the wheels with trusted publishing (environment `pypi`, no token).
   3. **release**: creates the GitHub pre-release with the same wheels.
4. Check: `pip install irradiance==0.0.3` in a fresh venv.

## Rules

- **PyPI versions are permanent.** A version can't be re-uploaded, even after
  deleting it. A bad release is *yanked* on pypi.org and fixed in a new
  version.
- **Wheels only, no sdist**, until the `irradiance` crate dependency comes
  from crates.io instead of a local path (it needs a release with `demosaic`).
- Release from `main` only, and only through the tag.

## When something fails

- **build fails**: nothing was published. Fix on `main`, then delete the tag
  (`git push origin :refs/tags/v0.0.3 && git tag -d v0.0.3`) and release again
  with the same version.
- **pypi fails** (e.g. the trusted publisher config changed): nothing was
  published. Fix and re-run the failed job from the Actions tab if the fix
  was outside the repo; otherwise delete the tag and release again.
- **release fails after pypi succeeded**: PyPI already has the version, so
  don't re-tag. Create the GitHub release from the run's wheels:

  ```sh
  gh run download <run-id> -p 'wheel-*' -D /tmp/wheels
  gh release create v0.0.3 /tmp/wheels/*/*.whl --prerelease --title v0.0.3 --generate-notes --verify-tag
  ```

## Moving to a new upstream `irradiance`

The upstream commit CI builds against is in `.github/irradiance-pin`. To move:
update your local checkout, put the new SHA in that file, run `just build ci`,
regenerate goldens only if an output change is intended (`just golden-update`,
and say why in the commit), then push and release as above. The weekly CI run
against upstream `main` shows whether a bump is safe before you make it.

## One-time setup (done)

The PyPI project has a trusted publisher: owner `jysf`, repository
`pyirradiance`, workflow `wheels.yml`, environment `pypi`.
