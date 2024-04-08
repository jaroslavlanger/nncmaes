check () {
    python -m ruff check --fix $1
    python -m mypy $1
    python -m pyright $1
    python -m doctest $1
}

format () {
    python -m ruff format $@
}
