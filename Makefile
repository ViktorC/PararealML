
install:
	pip3 install -r requirements.txt

lint:
	flake8 pararealml/ tests/

type-check:
	mypy --ignore-missing-imports pararealml/

test:
	pytest -sv tests/

coverage:
	coverage run --source=pararealml/core -m pytest -sv tests/
	coverage xml -o coverage.xml

run:
	mpiexec -n 4 python -m mpi4py -m examples.$(example)