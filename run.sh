#!/usr/bin/env bash
# Запуск Best-Face на macOS и Linux.
#   ./run.sh "/путь/к/папке/с/фото" [--limit N] [--output папка] [--no-contact-sheet]
# Без аргументов откроется диалог выбора папки (macOS/zenity/kdialog) или запрос в консоли.
# При первом запуске создаётся окружение .venv и ставятся зависимости.
set -u
cd "$(dirname "$0")" || exit 1
export PYTHONIOENCODING=utf-8

pause() {
    if [ -t 0 ] && [ -z "${BEST_FACE_NO_PAUSE:-}" ]; then
        read -r -n 1 -p "Нажмите любую клавишу для выхода..." _ || true
        echo
    fi
}

# ── Окружение ────────────────────────────────────────────────────────────────
PY="$PWD/.venv/bin/python"
if [ ! -f .venv/.deps_ok ]; then
    BASE_PY="$(command -v python3 || command -v python || true)"
    if [ -z "$BASE_PY" ]; then
        echo "Python 3.10+ не найден. Установите его (macOS: brew install python; Linux: пакет python3 + python3-venv)."
        pause; exit 1
    fi
    echo "Первый запуск: создаю окружение .venv и ставлю зависимости (несколько минут)..."
    if ! "$BASE_PY" -m venv .venv || ! "$PY" -m pip install --upgrade pip \
        || ! "$PY" -m pip install -r requirements.txt; then
        echo "Не удалось установить зависимости."
        pause; exit 1
    fi
    touch .venv/.deps_ok
fi

# ── Папка с фото ─────────────────────────────────────────────────────────────
SRC="${1:-}"
if [ $# -gt 0 ]; then shift; fi

if [ -z "$SRC" ]; then
    if [ "$(uname)" = "Darwin" ]; then
        SRC="$(osascript -e 'POSIX path of (choose folder with prompt "Папка с фотографиями")' 2>/dev/null || true)"
    elif command -v zenity >/dev/null 2>&1; then
        SRC="$(zenity --file-selection --directory --title="Папка с фотографиями" 2>/dev/null || true)"
    elif command -v kdialog >/dev/null 2>&1; then
        SRC="$(kdialog --getexistingdirectory "$HOME" 2>/dev/null || true)"
    fi
fi
if [ -z "$SRC" ]; then
    read -r -p "Путь к папке с фотографиями: " SRC
fi

SRC="${SRC/#\~/$HOME}"
if [ "${#SRC}" -gt 1 ]; then SRC="${SRC%/}"; fi

if [ ! -d "$SRC" ]; then
    echo "Это не папка: $SRC"
    pause; exit 1
fi

# ── Запуск ───────────────────────────────────────────────────────────────────
"$PY" best_face.py "$SRC" "$@"
RC=$?

echo
if [ "$RC" -eq 0 ]; then
    OUT="$SRC/best_face_output/best"
    if [ -d "$OUT" ]; then
        echo "Готово. Лучшие фото: $OUT"
        if [ "$(uname)" = "Darwin" ]; then open "$OUT"
        elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$OUT" >/dev/null 2>&1 || true
        fi
    fi
else
    echo "Завершено с ошибкой, код $RC."
fi
pause
exit "$RC"
