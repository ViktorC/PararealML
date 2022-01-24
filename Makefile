
install:
	pip3 install -r requirements.txt

lint:
	flake8 pararealml/ tests/

type-check:
	mypy --ignore-missing-imports --no-strict-optional pararealml/

test:
	pytest -sv tests/

coverage:
	coverage run --source=pararealml/ -m pytest -sv tests/
	coverage xml -o coverage.xml

run:
	mpiexec -n 4 python -m mpi4py -m examples.$(example)