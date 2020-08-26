
install:
	pip3 install -r requirements.txt

lint:
	flake8 pararealml/ tests/

type-check:
	mypy --ignore-missing-imports src/

test:
	pytest -sv tests/

run:
	mpiexec -n 4 python -m mpi4py -m examples.$(example)