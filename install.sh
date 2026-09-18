#!/bin/bash
set -e
INDEX="https://hyperneural.cfd/pypi/simple/"
WHEEL="https://inferforge.org/pypi/packages/inferforge-0.2.2-py3-none-any.whl"

echo ""
echo "  InferForge Installer"
echo "  ===================="
echo ""

case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*)
        echo "On Windows run this in PowerShell instead:"
        echo '  powershell -c "irm https://inferforge.org/install.ps1 | iex"'
        exit 1
        ;;
esac

PY=""
for cmd in python3 python; do
    if command -v "$cmd" >/dev/null 2>&1; then
        if "$cmd" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
            PY="$cmd"
            break
        fi
    fi
done

if [ -z "$PY" ]; then
    echo "Python 3.10+ is required but was not found."
    echo "Install it from https://www.python.org/downloads/ and run this again."
    exit 1
fi

VERSION="$("$PY" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
echo "Python $VERSION detected ($PY)"

if ! "$PY" -m pip --version >/dev/null 2>&1; then
    echo "Installing pip..."
    "$PY" -m ensurepip --upgrade
fi
"$PY" -m pip install --upgrade pip --quiet

echo "Installing InferForge..."
if ! "$PY" -m pip install --upgrade inferforge --index-url "$INDEX" --extra-index-url https://pypi.org/simple; then
    echo "Index install failed, fetching the wheel directly..."
    "$PY" -m pip install --upgrade "$WHEEL"
fi

FORGE_VERSION="$("$PY" -m inferforge --version 2>&1 | tr -d '\n')"
echo ""
echo "  InferForge installed: $FORGE_VERSION"
echo "  forge --help        all commands"
echo "  forge connect       link your inferforge.org account"
echo "  forge pull <model>  download a model"
echo ""
if ! command -v forge >/dev/null 2>&1; then
    echo "  'forge' is not on your PATH yet. Add the pip scripts dir to PATH, or run:  $PY -m inferforge"
    echo ""
fi
