if ! python -c 'from sys import version_info as v; exit(0 if v.major == 3 and v.minor >= 9 else 1)'; then
    echo "Error: The project was tested with Python 3.9, your version is: $(python --version)" >&2
    exit 1
fi
if ! python -m venv prod; then
    echo "Couldn't create the virtual environment!" >&2
    exit 2
fi
if ! prod/bin/pip install -r requirements/prod.txt; then
    echo "Installing python packages from requirements/prod.txt failed!" >&2
    exit 3
fi
if ! wget "https://github.com/numbbo/coco/archive/refs/tags/v2.6.3.zip"; then
    echo "Failed to download the numbbo/coco v2.6.3!" >&2
    exit 4
fi
if ! unzip v2.6.3.zip; then
    echo "Failed to unzip the numbbo/coco v2.6.3.zip" >&2
    exit 5
fi
rm v2.6.3.zip
prod/bin/python coco-2.6.3/do.py run-python
