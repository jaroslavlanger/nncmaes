check () {
    (
        if [ $# -ne 1 ]; then
            echo "Wrong number of arguments! Use: 'check <module>'." >&2
            exit 1
        fi
        python -m compileall -q $1 \
        && python -m ruff check --fix $1 \
        && python -m mypy $1 \
        && python -m pyright $1 \
        && python -O -m doctest $1
    )
}

format () {
    python -m ruff format $@
}
