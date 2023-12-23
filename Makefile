install:
	pip install -r requirements.txt
	pip install llama-cpp-python
	pip install -r requirements-dev.txt

fmt:
#	cd src && isort .
	cd src && black . -v

run:
	cd src && python3 ./backend.py

lint:
	flake8 src

push:
	while ! git push; do sleep 60; done

pull:
	while true; do git fetch; git pull; sleep 600; done
