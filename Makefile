setup:
	pip install -r requirements.txt

test:
	py.test test.py

display:
	python display.py $(file_name)

clean:
	rm -rf __pycache__

.PHONY: setup test clean
