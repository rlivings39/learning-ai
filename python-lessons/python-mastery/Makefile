test:
	python -m unittest discover -v -s ..

package:
	python setup.py sdist

format:
	isort . --extend-skip Solutions/ --extend-skip Data/; black . --extend-exclude '(Solutions/|Data/)'
