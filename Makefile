.PHONY: help install bump-patch bump-minor bump-major release clean

help:
	@echo "LayerZero Release Management"
	@echo ""
	@echo "Available commands:"
	@echo "  make install      - Install bump2version"
	@echo "  make bump-patch   - Bump patch version (0.1.3 -> 0.1.4)"
	@echo "  make bump-minor   - Bump minor version (0.1.3 -> 0.2.0)"
	@echo "  make bump-major   - Bump major version (0.1.3 -> 1.0.0)"
	@echo "  make release      - Push tags to trigger release"
	@echo "  make clean        - Clean build artifacts"

install:
	brew install bumpversion

bump-patch:
	bumpversion patch

bump-minor:
	bumpversion minor

bump-major:
	bumpversion major

release:
	git push
	git push --tags

clean:
	rm -rf dist/ build/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

