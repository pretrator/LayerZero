# Release Workflow Explained

## Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Developer runs bump command                         │
│                                                              │
│   $ make bump-patch                                         │
│   or                                                        │
│   $ bumpversion patch                                       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: bumpversion does automatically:                     │
│                                                              │
│   1. Updates pyproject.toml:                                │
│      version = "0.1.3" → version = "0.1.4"                 │
│                                                              │
│   2. Updates LayerZero/__init__.py:                         │
│      __version__ = "0.1.3" → __version__ = "0.1.4"         │
│                                                              │
│   3. Creates git commit:                                    │
│      "Bump version: 0.1.3 → 0.1.4"                         │
│                                                              │
│   4. Creates git tag:                                       │
│      v0.1.4                                                 │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Developer pushes to GitHub                          │
│                                                              │
│   $ make release                                            │
│   or                                                        │
│   $ git push && git push --tags                             │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 4: GitHub Actions triggers (on tag push)               │
│                                                              │
│   Workflow: .github/workflows/release.yml                   │
│   Trigger: on push of tag matching v*.*.*                   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 5: GitHub Actions builds package                       │
│                                                              │
│   1. Checks out code                                        │
│   2. Sets up Python 3.10                                    │
│   3. Installs build tools                                   │
│   4. Runs: python -m build                                  │
│      Creates: dist/LayerZero-0.1.4.tar.gz                   │
│                dist/LayerZero-0.1.4-py3-none-any.whl        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 6: GitHub Actions publishes to PyPI                    │
│                                                              │
│   Uses: PYPI_API_KEY secret                                 │
│   Runs: python -m twine upload dist/*                       │
│   Result: Package available on PyPI                         │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 7: GitHub Actions creates GitHub Release               │
│                                                              │
│   Creates release with auto-generated notes                 │
│   Attaches artifacts (wheel, tar.gz)                        │
└─────────────────────────────────────────────────────────────┘
```

---

## Detailed Example

### Before: Current State
```
Version: 0.1.3
Git: Clean working directory
```

### Command: Bump Patch Version
```bash
$ make bump-patch
```

### What Happens Locally

1. **File Updates:**
   ```diff
   # pyproject.toml
   - version = "0.1.3"
   + version = "0.1.4"
   
   # LayerZero/__init__.py
   - __version__ = "0.1.3"
   + __version__ = "0.1.4"
   ```

2. **Git Commit:**
   ```
   commit abc123def456
   Author: You <you@email.com>
   Date: Thu Oct 24 2025
   
       Bump version: 0.1.3 → 0.1.4
   ```

3. **Git Tag:**
   ```
   tag: v0.1.4
   ```

### Command: Push to GitHub
```bash
$ make release
# or
$ git push && git push --tags
```

### What Happens on GitHub

1. **GitHub detects tag push:** `v0.1.4`

2. **GitHub Actions starts:**
   ```
   Run #42: Release workflow
   Triggered by: tag push (v0.1.4)
   Status: Running...
   ```

3. **Build step:**
   ```bash
   ✓ Checkout code
   ✓ Setup Python 3.10
   ✓ Install build tools
   ✓ Build package
     Created dist/LayerZero-0.1.4.tar.gz
     Created dist/LayerZero-0.1.4-py3-none-any.whl
   ```

4. **Publish step:**
   ```bash
   ✓ Authenticate with PyPI (using PYPI_API_KEY)
   ✓ Upload dist/LayerZero-0.1.4.tar.gz
   ✓ Upload dist/LayerZero-0.1.4-py3-none-any.whl
   
   Package published: https://pypi.org/project/LayerZero/0.1.4/
   ```

5. **Release creation:**
   ```
   ✓ GitHub Release created
     Title: v0.1.4
     Tag: v0.1.4
     Auto-generated release notes
   ```

### After: Result

Users can now install:
```bash
pip install LayerZero==0.1.4
# or
pip install LayerZero  # Gets latest (0.1.4)
```

---

## Version Bump Types

### Patch (Bug fixes)
```bash
make bump-patch
# 0.1.3 → 0.1.4
# 0.1.4 → 0.1.5
# 1.2.9 → 1.2.10
```

### Minor (New features)
```bash
make bump-minor
# 0.1.3 → 0.2.0
# 1.2.9 → 1.3.0
```

### Major (Breaking changes)
```bash
make bump-major
# 0.1.3 → 1.0.0
# 1.2.9 → 2.0.0
```

---

## Prerequisites

### 1. Clean Git Working Directory

bumpversion requires no uncommitted changes:

```bash
# Check status
git status

# If dirty, commit changes first
git add .
git commit -m "Your changes"

# Then bump version
make bump-patch
```

### 2. PyPI API Token (One-time setup)

**Get token:**
1. Go to https://pypi.org/manage/account/token/
2. Click "Add API token"
3. Name: "LayerZero GitHub Actions"
4. Scope: "Entire account" or specific to LayerZero project
5. Copy the token (starts with `pypi-...`)

**Add to GitHub:**
1. Go to https://github.com/YOUR_USERNAME/LayerZero/settings/secrets/actions
2. Click "New repository secret"
3. Name: `PYPI_API_KEY`
4. Value: Paste your token
5. Click "Add secret"

---

## Testing the Workflow

### Test Locally (Dry Run)
```bash
# See what would happen without making changes
bumpversion --dry-run --verbose patch

# Output shows:
# - Current version: 0.1.3
# - New version: 0.1.4
# - Files that would be updated
# - Commit message
# - Tag name
```

### Test GitHub Actions

Option 1: Use a different branch
```bash
git checkout -b test-release
make bump-patch
make release
# Watch GitHub Actions run
# Won't affect main branch
```

Option 2: Use TestPyPI first

Modify `.github/workflows/release.yml`:
```yaml
# Change line 32 to:
run: python -m twine upload --repository testpypi dist/*
```

Then release normally. It will publish to TestPyPI instead.

---

## Troubleshooting

### "Working directory is dirty"
```bash
# Commit your changes first
git add .
git commit -m "Changes"
make bump-patch
```

### "Tag already exists"
```bash
# Delete tag locally and remotely
git tag -d v0.1.4
git push origin :refs/tags/v0.1.4

# Then try again
make bump-patch
make release
```

### "PyPI upload failed"
```bash
# Check GitHub secrets
# Settings → Secrets → Actions → PYPI_API_KEY

# Regenerate token if needed
# Re-add to GitHub secrets
```

### "Package already exists on PyPI"
```bash
# You can't overwrite PyPI versions
# Must bump to a new version
make bump-patch  # Try next version
make release
```

---

## Complete Release Checklist

- [ ] All changes committed
- [ ] Tests passing (if you have tests)
- [ ] README updated
- [ ] Git working directory clean
- [ ] PYPI_API_KEY secret configured in GitHub
- [ ] Run `make bump-patch` (or minor/major)
- [ ] Verify version updated in pyproject.toml and __init__.py
- [ ] Run `make release` to push
- [ ] Watch GitHub Actions run
- [ ] Verify package on PyPI
- [ ] Test installation: `pip install LayerZero==<new_version>`

---

## Quick Commands Reference

```bash
# Install bumpversion
brew install bumpversion

# Bump version (choose one)
make bump-patch    # Bug fixes: 0.1.3 → 0.1.4
make bump-minor    # Features:  0.1.3 → 0.2.0
make bump-major    # Breaking:  0.1.3 → 1.0.0

# Push and trigger release
make release       # Pushes code + tags

# Or manual
bumpversion patch
git push && git push --tags

# Dry run (test without changes)
bumpversion --dry-run patch

# Clean build artifacts
make clean
```

---

## Summary

**One command to release:**
```bash
make bump-patch && make release
```

That's it! Everything else is automatic.

