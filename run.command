#!/usr/bin/env bash
# macOS: двойной клик в Finder (или перетащите папку на этот файл).
exec "$(dirname "$0")/run.sh" "$@"
