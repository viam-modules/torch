PYTHONPATH := ./torch:$(PYTHONPATH)

.PHONY: test lint setup dist

test:
	PYTHONPATH=$(PYTHONPATH) pytest src/
lint:
	pylint --disable=E1101,W0719,C0202,R0801,W0613,C0411 src/
setup:
	python3 -m pip install -r requirements.txt -U
dist/archive.tar.gz:
	tar -czf module.tar.gz run.sh requirements.txt src