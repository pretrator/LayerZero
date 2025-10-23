# Release Guide

## Setup (One-time)

1. Install bumpversion:
```bash
brew install bumpversion
# or
make install
```

2. Add PyPI API token to GitHub:
   - Go to https://pypi.org/manage/account/token/
   - Create a new token
   - Add to GitHub: Settings → Secrets → Actions → New secret
   - Name: `PYPI_API_KEY`
   - Value: Your PyPI token

## Release Process

### Option 1: Using Makefile (Recommended)

```bash
# For bug fixes (0.1.3 -> 0.1.4)
make bump-patch

# For new features (0.1.3 -> 0.2.0)
make bump-minor

# For breaking changes (0.1.3 -> 1.0.0)
make bump-major

# Push to trigger release
make release
```

### Option 2: Manual bumpversion

```bash
# Bump version (choose one)
bumpversion patch   # 0.1.3 -> 0.1.4
bumpversion minor   # 0.1.3 -> 0.2.0
bumpversion major   # 0.1.3 -> 1.0.0

# Push changes and tag
git push
git push --tags
```

## What Happens Automatically

1. **bumpversion** updates:
   - `pyproject.toml` version
   - `LayerZero/__init__.py` version
   - Creates git commit
   - Creates git tag (e.g., `v0.1.4`)

2. **GitHub Actions** (on tag push):
   - Builds the package
   - Publishes to PyPI
   - Creates GitHub release with notes

## Version Scheme

- **Patch** (0.1.X): Bug fixes, small improvements
- **Minor** (0.X.0): New features, backward compatible
- **Major** (X.0.0): Breaking changes

## Manual Release (Without bump2version)

If you prefer manual control:

```bash
# 1. Update version in pyproject.toml and __init__.py
# 2. Commit changes
git add .
git commit -m "Release v0.1.4"

# 3. Create and push tag
git tag v0.1.4
git push origin v0.1.4
```

## Troubleshooting

### "Tag already exists"
```bash
git tag -d v0.1.4          # Delete local tag
git push origin :v0.1.4    # Delete remote tag
```

### "PyPI upload failed"
Check that `PYPI_API_KEY` secret is set correctly in GitHub.

### Test release first
To test without publishing to PyPI, remove the tag:
```bash
git push origin :v0.1.4
```

