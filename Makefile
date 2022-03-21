all: clean test test-nojit docs

clean:
	rm -rf docs/
	rm -rf dist/
	rm -rf gps_frames.egg-info/
	black gps_frames/

build:
	python setup.py sdist

docs:
	pdoc --html --output-dir docs --template-dir templates  gps_frames --force
	mv docs/gps_frames/* docs/
	rm -r docs/gps_frames

test:
	python -m pytest --cov=gps_frames --cov-report term-missing tests

test-nojit:
	NUMBA_DISABLE_JIT=1 python -m pytest --cov=gps_frames --cov-report term-missing tests

example:
	python example.py