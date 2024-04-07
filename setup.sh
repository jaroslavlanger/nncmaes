if ! python3 -c 'from sys import version_info as v; exit(0 if v.major >= 3 and v.minor >= 9 else 1)'; then
    echo "Error: The project was tested with Python 3.9, you have $(python3 --version)" >&2
    exit 42
fi
sudo apt install build-essential python3-dev unzip
python3 -m venv venv
venv/bin/pip install -r requirements/prod.txt
wget "https://github.com/numbbo/coco/archive/refs/tags/v2.6.3.zip"
unzip v2.6.3.zip
cd coco-2.6.3
../venv/bin/python do.py run-python
