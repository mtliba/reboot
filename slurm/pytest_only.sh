#!/bin/bash
export LC_ALL=C; cd ${REBOOT_ROOT:-$PWD}
find reboot scripts tests -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null
PYTHONPATH=. timeout 900 ${REBOOT_PYTHON:-python} -m pytest -q -p no:warnings 2>&1 | tail -3
