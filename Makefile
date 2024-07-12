PYTHONPATH := ./torch:$(PYTHONPATH)

.PHONY: test lint build dist

test:
	PYTHONPATH=$(PYTHONPATH) pytest src/
lint:
	pylint --disable=E1101,W0719,C0202,R0801,W0613,C0411 src/
build:
	./build.sh
dist/archive.tar.gz:
	tar -czvf dist/archive.tar.gz dist/main