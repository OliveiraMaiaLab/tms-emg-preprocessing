## 5. Regenerating the environment files

After installing or upgrading packages, refresh the specs from the **activated**
environment and commit the results:

```bash
# Portable Conda spec (versions, no build hashes) — cross-platform
conda env export --no-builds | findstr -v "prefix:" > environment.yml      # Windows
conda env export --no-builds | grep -v "^prefix:" > environment.yml        # macOS/Linux

# Exact, build-locked Windows clone (run on the win-64 machine)
conda list --explicit > conda-win64-explicit.txt

# Clean pip pins (package==version, no local file:/// paths)
python -m pip list --format=freeze > requirements-pip.txt
```

> `conda env export --no-builds` produces a *full* dump (all transitive
> packages) tied to the current OS. The `environment.yml` committed here is a
> hand-curated subset for portability — when refreshing it, prefer editing the
> curated file over overwriting it with the full dump, or keep the full dump as
> a separate `environment-full.yml`.
>
> Use `python -m pip list --format=freeze` (not `pip freeze`): in a Conda env,
> plain `pip freeze` emits unusable `@ file:///C:/...` local paths.
