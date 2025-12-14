# Contributing to FrEVL

Thank you for your interest in contributing to FrEVL! This document provides guidelines and instructions for contributing to the project.

## Getting Started

### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/FrEVL.git
   cd FrEVL
   ```

2. **Create a virtual environment**
   ```bash
   conda create -n frevl-dev python=3.9
   conda activate frevl-dev
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

4. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

5. **Run tests to verify setup**
   ```bash
   pytest tests/
   ```

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-number-description
```

### 2. Code Quality Checks

Run these before committing:

```bash
# Format code
make format

# Run linters
make lint

# Run tests
make test

# Check type hints
make typecheck
```

### 3. Commit Your Changes

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```bash
git commit -m "feat: add new fusion attention mechanism"
git commit -m "fix: resolve memory leak in batch processing"
git commit -m "docs: update installation instructions"
```

Commit types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

### 4. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:
- Clear title and description
- Link to related issues
- Screenshots/examples if applicable
- Test results

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_model.py

# Run with coverage
pytest tests/ --cov=frevl --cov-report=html

# Run specific test
pytest tests/test_model.py::TestFrEVL::test_forward_pass
```

### Writing Tests

Create test files in `tests/` following the pattern `test_*.py`:

```python
import pytest
import torch
from frevl.model import FrEVL

class TestFrEVL:
    def test_forward_pass(self):
        model = FrEVL()
        # Your test code
        assert output.shape == expected_shape
    
    def test_attention_weights(self):
        # Test attention mechanism
        pass
```

## Documentation

### Docstring Format

We use Google-style docstrings:

```python
def process_image(image: np.ndarray, size: Tuple[int, int]) -> torch.Tensor:
    """
    Process image for model input.
    
    Args:
        image: Input image as numpy array
        size: Target size (height, width)
    
    Returns:
        Processed image tensor
    
    Raises:
        ValueError: If image dimensions are invalid
    
    Example:
        >>> img = process_image(np_array, (224, 224))
    """
    pass
```

### Updating Documentation

1. API documentation is auto-generated from docstrings
2. Update README.md for user-facing changes
3. Add examples in `examples/` for new features
4. Update CHANGELOG.md following [Keep a Changelog](https://keepachangelog.com/)

## Reporting Issues

### Before Submitting an Issue

1. Check existing issues to avoid duplicates
2. Try the latest version
3. Verify your environment setup

### Issue Template

```markdown
## Description
Brief description of the issue

## Steps to Reproduce
1. Step one
2. Step two
3. ...

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- Python version:
- PyTorch version:
- CUDA version:
- OS:

## Additional Context
Any other relevant information
```

## Areas for Contribution

### Good First Issues

Look for issues tagged with `good first issue`:
- Documentation improvements
- Adding unit tests
- Simple bug fixes
- Code formatting

### Priority Areas

- **Performance Optimization**: Inference speed improvements
- **Model Variants**: New architectures or task adaptations
- **Datasets**: Support for additional datasets
- **Deployment**: Cloud deployment templates
- **Monitoring**: Better metrics and logging
- **Documentation**: Tutorials and examples

## Pull Request Process

### PR Checklist

- [ ] Code follows project style guidelines
- [ ] Tests pass locally
- [ ] Documentation is updated
- [ ] Commits follow conventional format
- [ ] PR description clearly explains changes
- [ ] Related issues are linked
- [ ] No merge conflicts

### Review Process

1. Automated checks must pass
2. Code review by maintainer
3. Address review feedback
4. Approval and merge

## Recognition

Contributors will be:
- Added to CONTRIBUTORS.md
- Mentioned in release notes
- Given credit in relevant documentation

## Style Guide

### Python Style

- Follow PEP 8
- Maximum line length: 100 characters
- Use type hints for function signatures
- Prefer f-strings for formatting

### Import Order

```python
# Standard library
import os
import sys

# Third-party
import torch
import numpy as np

# Local
from frevl.model import FrEVL
from frevl.utils import load_checkpoint
```

## Development Tools

### Makefile Commands

```bash
make help        # Show all commands
make setup       # Setup development environment
make format      # Format code
make lint        # Run linters
make test        # Run tests
make docs        # Build documentation
make clean       # Clean build artifacts
make release     # Prepare release
```

### Pre-commit Hooks

Our `.pre-commit-config.yaml` runs:
- Black (code formatting)
- isort (import sorting)
- Flake8 (linting)
- MyPy (type checking)

## Performance Guidelines

When contributing performance improvements:

1. **Benchmark Before and After**
   ```bash
   python benchmarks/benchmark_inference.py --before
   # Make changes
   python benchmarks/benchmark_inference.py --after
   ```

2. **Profile Code**
   ```python
   from frevl.utils import profile_memory
   
   @profile_memory
   def your_function():
       pass
   ```

3. **Document Performance Gains**
   Include benchmark results in PR description

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to FrEVL! Your efforts help make vision-language understanding more efficient and accessible to everyone. 🚀
