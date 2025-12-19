all: clean test test-nojit docs

clean:
	rm -rf site/
	rm -rf dist/
	rm -rf gps_frames.egg-info/
	black gps_frames/

build:
	python -m build

docs:
	mkdocs build

test:
	python -m pytest --cov=gps_frames --cov-report term-missing tests

test-nojit:
	NUMBA_DISABLE_JIT=1 python -m pytest --cov=gps_frames --cov-report term-missing tests

example:
	python examples/example.py