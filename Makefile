setup:
	python setup.py install

package:
	python setup.py sdist bdist_wheel

publish-test:
	python -m twine upload --repository testpypi dist/*

publish:
	python -m twine upload dist/*

install:
	pip3 install -r requirements.txt

lint:
	flake8 pararealml/ tests/ examples/

type-check:
	mypy pararealml/ tests/ examples/

format:
	black pararealml/ tests/ examples/
	isort pararealml/ tests/ examples/

format-check:
	black --check pararealml/ tests/ examples/
	isort --check-only pararealml/ tests/ examples/

test:
	pytest -v tests/

coverage:
	coverage run --source=pararealml/ -m pytest -v tests/
	coverage xml -o coverage.xml

run:
	mpiexec -n $(p) python -m mpi4py -m examples.$(example)