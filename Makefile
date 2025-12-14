# FrEVL Makefile
# Automated development and deployment tasks

.PHONY: help setup format lint test clean docker deploy docs

# Variables
PYTHON := python3
PIP := pip
DOCKER := docker
DOCKER_COMPOSE := docker-compose
PROJECT_NAME := frevl
VERSION := $(shell $(PYTHON) -c "import frevl; print(frevl.__version__)" 2>/dev/null || echo "0.1.0")

# Colors for terminal output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Display help message
help:
	@echo "$(BLUE)FrEVL Development Commands$(NC)"
	@echo ""
	@echo "$(GREEN)Setup & Installation:$(NC)"
	@echo "  make setup          Install all dependencies and setup environment"
	@echo "  make setup-dev      Setup development environment with pre-commit hooks"
	@echo "  make download-data  Download required datasets"
	@echo ""
	@echo "$(GREEN)Development:$(NC)"
	@echo "  make format         Format code with black and isort"
	@echo "  make lint           Run all linters (flake8, pylint, mypy)"
	@echo "  make test           Run all tests with coverage"
	@echo "  make test-fast      Run fast unit tests only"
	@echo "  make benchmark      Run performance benchmarks"
	@echo ""
	@echo "$(GREEN)Model Operations:$(NC)"
	@echo "  make train          Train model with default config"
	@echo "  make evaluate       Evaluate model on validation set"
	@echo "  make demo           Launch interactive demo"
	@echo "  make serve          Start API server"
	@echo ""
	@echo "$(GREEN)Docker & Deployment:$(NC)"
	@echo "  make docker-build   Build Docker images"
	@echo "  make docker-run     Run Docker containers"
	@echo "  make docker-push    Push images to registry"
	@echo "  make deploy-k8s     Deploy to Kubernetes"
	@echo ""
	@echo "$(GREEN)Documentation:$(NC)"
	@echo "  make docs           Build documentation"
	@echo "  make docs-serve     Serve documentation locally"
	@echo ""
	@echo "$(GREEN)Maintenance:$(NC)"
	@echo "  make clean          Clean build artifacts and cache"
	@echo "  make clean-all      Deep clean including models and data"
	@echo "  make check-security Run security checks"
	@echo "  make release        Prepare new release"

#################################################################################
# SETUP & INSTALLATION                                                          #
#################################################################################

## Install all dependencies
setup:
	@echo "$(YELLOW)Setting up FrEVL environment...$(NC)"
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt
	@echo "$(GREEN)✓ Setup complete!$(NC)"

## Setup development environment
setup-dev: setup
	@echo "$(YELLOW)Setting up development environment...$(NC)"
	$(PIP) install -r requirements-dev.txt
	pre-commit install
	@echo "$(GREEN)✓ Development setup complete!$(NC)"

## Download datasets
download-data:
	@echo "$(YELLOW)Downloading datasets...$(NC)"
	$(PYTHON) scripts/download_datasets.py --dataset all --data-dir ./data
	@echo "$(GREEN)✓ Data download complete!$(NC)"

## Download pretrained models
download-models:
	@echo "$(YELLOW)Downloading pretrained models...$(NC)"
	$(PYTHON) scripts/download_models.py --model all
	@echo "$(GREEN)✓ Model download complete!$(NC)"

#################################################################################
# DEVELOPMENT                                                                    #
#################################################################################

## Format code
format:
	@echo "$(YELLOW)Formatting code...$(NC)"
	black .
	isort .
	@echo "$(GREEN)✓ Formatting complete!$(NC)"

## Run linters
lint:
	@echo "$(YELLOW)Running linters...$(NC)"
	flake8 . --count --statistics
	pylint frevl/ --fail-under=8.0
	mypy frevl/ --ignore-missing-imports
	@echo "$(GREEN)✓ Linting complete!$(NC)"

## Run all tests
test:
	@echo "$(YELLOW)Running tests...$(NC)"
	pytest tests/ -v --cov=frevl --cov-report=html --cov-report=term
	@echo "$(GREEN)✓ Tests complete! Coverage report: htmlcov/index.html$(NC)"

## Run fast tests only
test-fast:
	@echo "$(YELLOW)Running fast tests...$(NC)"
	pytest tests/ -v -m "not slow"
	@echo "$(GREEN)✓ Fast tests complete!$(NC)"

## Run specific test file
test-file:
	@echo "$(YELLOW)Running test file: $(FILE)$(NC)"
	pytest $(FILE) -v
	@echo "$(GREEN)✓ Test complete!$(NC)"

## Run benchmarks
benchmark:
	@echo "$(YELLOW)Running performance benchmarks...$(NC)"
	$(PYTHON) benchmarks/benchmark_inference.py --output results/benchmark_$(shell date +%Y%m%d_%H%M%S).json
	$(PYTHON) benchmarks/benchmark_training.py
	@echo "$(GREEN)✓ Benchmarks complete!$(NC)"

## Check code quality
quality: format lint test
	@echo "$(GREEN)✓ All quality checks passed!$(NC)"

#################################################################################
# MODEL OPERATIONS                                                              #
#################################################################################

## Train model
train:
	@echo "$(YELLOW)Training FrEVL model...$(NC)"
	$(PYTHON) train.py \
		--dataset vqa \
		--batch-size 128 \
		--epochs 20 \
		--learning-rate 1e-4 \
		--output-dir outputs/train_$(shell date +%Y%m%d_%H%M%S)
	@echo "$(GREEN)✓ Training complete!$(NC)"

## Evaluate model
evaluate:
	@echo "$(YELLOW)Evaluating model...$(NC)"
	$(PYTHON) evaluate.py \
		--model checkpoints/best_model.pt \
		--dataset all \
		--output-dir outputs/eval_$(shell date +%Y%m%d_%H%M%S)
	@echo "$(GREEN)✓ Evaluation complete!$(NC)"

## Launch demo
demo:
	@echo "$(YELLOW)Launching FrEVL demo...$(NC)"
	$(PYTHON) demo.py --model frevl-base --port 7860
	@echo "$(GREEN)✓ Demo running at http://localhost:7860$(NC)"

## Start API server
serve:
	@echo "$(YELLOW)Starting FrEVL API server...$(NC)"
	uvicorn serve:app --host 0.0.0.0 --port 8000 --reload
	@echo "$(GREEN)✓ API running at http://localhost:8000$(NC)"

## Export model to ONNX
export-onnx:
	@echo "$(YELLOW)Exporting model to ONNX...$(NC)"
	$(PYTHON) scripts/export_onnx.py \
		--model checkpoints/best_model.pt \
		--output checkpoints/model.onnx
	@echo "$(GREEN)✓ ONNX export complete!$(NC)"

#################################################################################
# DOCKER & DEPLOYMENT                                                           #
#################################################################################

## Build Docker images
docker-build:
	@echo "$(YELLOW)Building Docker images...$(NC)"
	$(DOCKER) build -t $(PROJECT_NAME):$(VERSION) .
	$(DOCKER) build -t $(PROJECT_NAME):latest .
	$(DOCKER) build -f Dockerfile.demo -t $(PROJECT_NAME)-demo:latest .
	@echo "$(GREEN)✓ Docker build complete!$(NC)"

## Run Docker containers
docker-run:
	@echo "$(YELLOW)Starting Docker containers...$(NC)"
	$(DOCKER_COMPOSE) up -d
	@echo "$(GREEN)✓ Containers running!$(NC)"
	@echo "API: http://localhost:8000"
	@echo "Demo: http://localhost:7860"
	@echo "Grafana: http://localhost:3000"

## Stop Docker containers
docker-stop:
	@echo "$(YELLOW)Stopping Docker containers...$(NC)"
	$(DOCKER_COMPOSE) down
	@echo "$(GREEN)✓ Containers stopped!$(NC)"

## Push Docker images to registry
docker-push:
	@echo "$(YELLOW)Pushing Docker images...$(NC)"
	$(DOCKER) push $(PROJECT_NAME):$(VERSION)
	$(DOCKER) push $(PROJECT_NAME):latest
	@echo "$(GREEN)✓ Images pushed to registry!$(NC)"

## Deploy to Kubernetes
deploy-k8s:
	@echo "$(YELLOW)Deploying to Kubernetes...$(NC)"
	kubectl apply -f deploy/k8s/
	kubectl rollout status deployment/frevl-api -n frevl-production
	@echo "$(GREEN)✓ Kubernetes deployment complete!$(NC)"

## Deploy to AWS
deploy-aws:
	@echo "$(YELLOW)Deploying to AWS...$(NC)"
	$(PYTHON) scripts/deploy_aws.py
	@echo "$(GREEN)✓ AWS deployment complete!$(NC)"

#################################################################################
# DOCUMENTATION                                                                 #
#################################################################################

## Build documentation
docs:
	@echo "$(YELLOW)Building documentation...$(NC)"
	cd docs && make html
	@echo "$(GREEN)✓ Documentation built! Open docs/_build/html/index.html$(NC)"

## Serve documentation locally
docs-serve: docs
	@echo "$(YELLOW)Serving documentation...$(NC)"
	cd docs/_build/html && $(PYTHON) -m http.server 8080
	@echo "$(GREEN)✓ Documentation at http://localhost:8080$(NC)"

## Generate API documentation
docs-api:
	@echo "$(YELLOW)Generating API documentation...$(NC)"
	sphinx-apidoc -o docs/api frevl/
	@echo "$(GREEN)✓ API documentation generated!$(NC)"

#################################################################################
# MAINTENANCE                                                                    #
#################################################################################

## Clean build artifacts
clean:
	@echo "$(YELLOW)Cleaning build artifacts...$(NC)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	@echo "$(GREEN)✓ Clean complete!$(NC)"

## Deep clean including models and data
clean-all: clean
	@echo "$(YELLOW)Deep cleaning...$(NC)"
	rm -rf data/
	rm -rf checkpoints/
	rm -rf outputs/
	rm -rf logs/
	rm -rf cache/
	@echo "$(GREEN)✓ Deep clean complete!$(NC)"

## Run security checks
check-security:
	@echo "$(YELLOW)Running security checks...$(NC)"
	bandit -r frevl/
	safety check
	@echo "$(GREEN)✓ Security checks complete!$(NC)"

## Check for dependency updates
check-updates:
	@echo "$(YELLOW)Checking for dependency updates...$(NC)"
	pip list --outdated
	@echo "$(GREEN)✓ Update check complete!$(NC)"

## Create release
release:
	@echo "$(YELLOW)Preparing release $(VERSION)...$(NC)"
	@echo "1. Updating version..."
	$(PYTHON) scripts/update_version.py $(VERSION)
	@echo "2. Running tests..."
	make test
	@echo "3. Building package..."
	$(PYTHON) setup.py sdist bdist_wheel
	@echo "4. Creating git tag..."
	git tag -a v$(VERSION) -m "Release version $(VERSION)"
	@echo "$(GREEN)✓ Release prepared! Run 'make publish' to upload to PyPI$(NC)"

## Publish to PyPI
publish:
	@echo "$(YELLOW)Publishing to PyPI...$(NC)"
	twine upload dist/*
	@echo "$(GREEN)✓ Published to PyPI!$(NC)"

#################################################################################
# MONITORING & PROFILING                                                        #
#################################################################################

## Profile memory usage
profile-memory:
	@echo "$(YELLOW)Profiling memory usage...$(NC)"
	$(PYTHON) -m memory_profiler train.py --epochs 1
	@echo "$(GREEN)✓ Memory profiling complete!$(NC)"

## Profile CPU usage
profile-cpu:
	@echo "$(YELLOW)Profiling CPU usage...$(NC)"
	$(PYTHON) -m cProfile -o profile.stats train.py --epochs 1
	$(PYTHON) -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)"
	@echo "$(GREEN)✓ CPU profiling complete!$(NC)"

## Monitor GPU usage
monitor-gpu:
	@echo "$(YELLOW)Monitoring GPU usage...$(NC)"
	nvidia-smi --loop=1

## Start monitoring stack
monitor-start:
	@echo "$(YELLOW)Starting monitoring stack...$(NC)"
	$(DOCKER_COMPOSE) -f docker-compose.monitoring.yml up -d
	@echo "$(GREEN)✓ Monitoring stack running!$(NC)"
	@echo "Prometheus: http://localhost:9090"
	@echo "Grafana: http://localhost:3000"

#################################################################################
# CI/CD                                                                         #
#################################################################################

## Run CI pipeline locally
ci-local:
	@echo "$(YELLOW)Running CI pipeline locally...$(NC)"
	act -j test
	@echo "$(GREEN)✓ CI pipeline complete!$(NC)"

## Validate GitHub Actions
ci-validate:
	@echo "$(YELLOW)Validating GitHub Actions...$(NC)"
	actionlint .github/workflows/*.yml
	@echo "$(GREEN)✓ GitHub Actions valid!$(NC)"

#################################################################################
# SHORTCUTS                                                                     #
#################################################################################

# Shortcuts for common tasks
fmt: format
t: test
tf: test-fast
b: benchmark
d: demo
s: serve
r: docker-run
