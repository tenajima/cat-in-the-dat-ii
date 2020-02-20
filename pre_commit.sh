poetry run black scripts
poetry run isort scripts/*.py
poetry run flake8 --ignore=E501,W503 scripts