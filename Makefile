SHELL := /bin/bash

.PHONY: install
install:
	pip install -r requirements.txt

run-server:
	uvicorn backend.app:app --host 0.0.0.0 --port 80 --reload

train:
	cd backend/ner_src && python train.py && cd -

build-docker:
	docker build -t ner-chrome-extension .

run-docker:
	docker run -p 8080:8080 ner-chrome-extension