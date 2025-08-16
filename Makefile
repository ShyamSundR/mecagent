# Makefile for CadQuery Code Generator Technical Test
# Provides convenient commands for the complete pipeline

.PHONY: help setup train-baseline train-enhanced infer eval clean test demo

# Default target
help:
	@echo "CadQuery Code Generator - Available Commands:"
	@echo ""
	@echo "Setup:"
	@echo "  make setup          - Install dependencies with uv"
	@echo ""
	@echo "Training:"
	@echo "  make train-baseline - Train baseline model"
	@echo "  make train-enhanced - Train enhanced model"
	@echo ""
	@echo "Inference:"
	@echo "  make infer          - Run inference on test images"
	@echo ""
	@echo "Evaluation:"
	@echo "  make eval           - Evaluate predictions"
	@echo ""
	@echo "Development:"
	@echo "  make test           - Run unit tests"
	@echo "  make demo           - Run quick demo"
	@echo "  make clean          - Clean generated files"
	@echo ""

# Setup environment
setup:
	@echo "Setting up environment..."
	uv sync
	@echo "Environment setup complete!"
	@echo "Activate with: source .venv/bin/activate"

# Train baseline model
train-baseline:
	@echo "Training baseline model..."
	python -m src.train \
		--config configs/baseline.yaml \
		--model_type baseline \
		--output_dir checkpoints/baseline \
		--mixed_precision

# Train enhanced model
train-enhanced:
	@echo "Training enhanced model..."
	python -m src.train \
		--config configs/enhanced.yaml \
		--model_type enhanced \
		--output_dir checkpoints/enhanced \
		--mixed_precision

# Run inference
infer:
	@echo "Running inference..."
	python -m src.infer \
		--checkpoint checkpoints/baseline/best_model.pt \
		--config configs/baseline.yaml \
		--model_type baseline \
		--images data/test_images \
		--output results/baseline_predictions.jsonl \
		--batch_size 4

# Run inference with enhanced model
infer-enhanced:
	@echo "Running enhanced inference..."
	python -m src.infer \
		--checkpoint checkpoints/enhanced/best_model.pt \
		--config configs/enhanced.yaml \
		--model_type enhanced \
		--images data/test_images \
		--output results/enhanced_predictions.jsonl \
		--batch_size 4 \
		--use_grammar \
		--use_execution_guidance \
		--use_reranking

# Evaluate predictions
eval:
	@echo "Evaluating predictions..."
	python -m src.eval \
		--predictions results/baseline_predictions.jsonl \
		--references data/test_references.jsonl \
		--output results/baseline_evaluation.json \
		--csv_output results/baseline_evaluation.csv

# Evaluate enhanced predictions
eval-enhanced:
	@echo "Evaluating enhanced predictions..."
	python -m src.eval \
		--predictions results/enhanced_predictions.jsonl \
		--references data/test_references.jsonl \
		--output results/enhanced_evaluation.json \
		--csv_output results/enhanced_evaluation.csv

# Run quick demo
demo:
	@echo "Running quick demo..."
	@python test_setup.py
	@echo "Demo completed successfully!"

# Run tests
test:
	@echo "Running tests..."
	python -m pytest tests/ -v

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	rm -rf checkpoints/
	rm -rf results/
	rm -rf __pycache__/
	rm -rf src/__pycache__/
	rm -rf src/models/__pycache__/
	rm -rf src/utils/__pycache__/
	rm -rf .pytest_cache/
	@echo "Clean complete!"

# Full pipeline (baseline)
pipeline-baseline: train-baseline infer eval
	@echo "Baseline pipeline completed!"

# Full pipeline (enhanced)
pipeline-enhanced: train-enhanced infer-enhanced eval-enhanced
	@echo "Enhanced pipeline completed!"

# Compare results
compare:
	@echo "Comparing baseline vs enhanced results..."
	@if [ -f results/baseline_evaluation.csv ] && [ -f results/enhanced_evaluation.csv ]; then \
		echo "Baseline Results:"; \
		cat results/baseline_evaluation.csv; \
		echo ""; \
		echo "Enhanced Results:"; \
		cat results/enhanced_evaluation.csv; \
	else \
		echo "Results files not found. Run 'make eval' and 'make eval-enhanced' first."; \
	fi

# Quick setup for development
dev-setup: setup
	@echo "Creating necessary directories..."
	mkdir -p data/test_images
	mkdir -p data/test_references
	mkdir -p checkpoints/baseline
	mkdir -p checkpoints/enhanced
	mkdir -p results
	@echo "Development setup complete!"

# Show model info
model-info:
	@echo "Model Information:"
	@python -c "import sys; sys.path.append('src'); print('Model info: Import successful')"
