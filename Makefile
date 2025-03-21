setup:
	./setup.sh

dist/archive.tar.gz: setup
	./build.sh
