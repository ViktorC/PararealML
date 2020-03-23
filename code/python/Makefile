
install:
	pip3 install -r requirements.txt

lint:
	flake8 src/

test:
	pytest -sv tests/

run:
	mpiexec -n 4 python -m mpi4py -m src.main