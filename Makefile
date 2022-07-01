setup:
	pip install -r requirements.txt

test:
	py.test test.py

clean:
	rm -rf __pycache__

.PHONY: setup test clean
