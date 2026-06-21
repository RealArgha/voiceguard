# VoiceGuard — GNU Make build file (requires Git-for-Windows make or choco install make)
.DEFAULT_GOAL := help

PYTHON  := .venv/Scripts/python
PIP     := .venv/Scripts/pip
UVICORN := .venv/Scripts/uvicorn

.PHONY: help run install finetune train quick-train test clean mobile-install

help:
	@echo "VoiceGuard — available targets:"
	@echo "  make run                 Start FastAPI server on :8000 (reload on)"
	@echo "  make install             Create .venv and install requirements"
	@echo "  make finetune            Fine-tune on data/mini/ (best settings)"
	@echo "  make train               Full 2019+2021 training (10 epochs)"
	@echo "  make quick-train         2-epoch sanity check (~5 min)"
	@echo "  make test FILE=clip.wav  Score a single audio file"
	@echo "  make clean               Remove cache dirs and .pyc files"
	@echo "  make mobile-install      npm install for voiceguard-mobile/"

run:
	$(UVICORN) backend.main:app --reload --port 8000

install:
	python -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

finetune:
	$(PYTHON) -m backend.finetune --no-augment --warmup 0 --lr 1e-4 --epochs 15

train:
	$(PYTHON) -m backend.train_2021 --augment --epochs 10 --batch-size 256

quick-train:
	$(PYTHON) -m backend.train_2021 --quick

test:
ifndef FILE
	$(error Usage: make test FILE=clip.wav)
endif
	$(PYTHON) test_model.py $(FILE)

clean:
	$(PYTHON) -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in ['cache','cache_audio']]; [p.unlink() for p in pathlib.Path('.').rglob('*.pyc')]"

mobile-install:
	cd voiceguard-mobile && npm install
