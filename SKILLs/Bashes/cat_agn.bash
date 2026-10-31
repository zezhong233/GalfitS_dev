#!/usr/bin/env bash
# cat_agn.bash -- append an AGN block (N{char}) to an existing galfitS lyric.
#
# Pure template: no python needed. Only the name and redshift are substituted;
# everything else is the default AGN parameter set.
#
# Usage:
#   bash cat_agn.bash --targfile xxx.lyric --name rubins_agn --zred 6.684 --char a
#
# Options:
#   --targfile   existing lyric file to append to (required)
#   --name       AGN component name, written to N{char}1) (required)
#   --char       single letter: AGN section index (required)
#   --zred       AGN redshift init, written to N{char}2) (required;
#                note the default bounds are [1,12], i.e. z>1 AGN)
#   --ebv        MW E(B-V) for N{char}3) (default 0)

set -euo pipefail

usage() { sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; }

TARGFILE="" ; NAME="" ; CHAR="" ; ZRED="" ; EBV="0"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --targfile) TARGFILE="$2"; shift 2 ;;
        --name)     NAME="$2";     shift 2 ;;
        --char)     CHAR="$2";     shift 2 ;;
        --zred)     ZRED="$2";     shift 2 ;;
        --ebv)      EBV="$2";      shift 2 ;;
        -h|--help)  usage; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

for v in TARGFILE NAME CHAR ZRED; do
    if [[ -z "${!v}" ]]; then
        echo "ERROR: --$(tr 'A-Z' 'a-z' <<< "$v") is required" >&2; usage; exit 1
    fi
done
[[ -f "$TARGFILE" ]] || { echo "ERROR: targfile not found: $TARGFILE" >&2; exit 1; }
[[ "$CHAR" =~ ^[a-z]$ ]] || { echo "ERROR: --char must be a single lowercase letter" >&2; exit 1; }
if grep -qE "^N${CHAR}[0-9]+\)" "$TARGFILE"; then
    echo "ERROR: AGN section N${CHAR}x) already exists in $TARGFILE" >&2; exit 1
fi

{
    echo ""
    echo "# AGN ${CHAR}: ${NAME} (z=${ZRED})"
    echo "N${CHAR}1) ${NAME}"
    echo "N${CHAR}2) [${ZRED}, 1,12,1,0]"
    echo "N${CHAR}3) ${EBV}"
    echo "N${CHAR}4) [0, -0.3, 0.3, 0.01, 1]"
    echo "N${CHAR}5) [0, -0.3, 0.3, 0.01, 1]"
    echo "N${CHAR}6) [7, 5, 10, 0.1, 0]"
    echo "N${CHAR}7) [-1, -4, 2, 0.1, 0]"
    echo "N${CHAR}8) [0, 0, 0.99, 0.01, 0]"
    echo "N${CHAR}9) [0, 0, 4, 0.25, 0]"
    echo "N${CHAR}10) [42, 41, 48, 0.1, 0]"
    echo "N${CHAR}11) [1, 0, 4, 0.1, 0]"
    echo "N${CHAR}12) []"
    echo "N${CHAR}13) []"
    echo "N${CHAR}14) 0"
    echo "N${CHAR}15) 0"
    echo "N${CHAR}16) 0"
    echo "N${CHAR}17) 0"
    echo "N${CHAR}18) 4"
    echo "N${CHAR}19) [1, 0.5, 2, 0.05, 0]"
    echo "N${CHAR}20) 0"
    echo "N${CHAR}21) [41, 39, 44, 0.1, 0]"
    echo "N${CHAR}22) [-0.5, -2.5, -0.25, 0.05, 0]"
    echo "N${CHAR}23) [0.5, 0.25, 1.5, 0.05, 0]"
    echo "N${CHAR}24) [7, 5, 10, 0.5, 0]"
    echo "N${CHAR}25) [15, 0, 90, 5, 0]"
    echo "N${CHAR}26) [1., 0.2, 5, 0.1, 0]"
    echo "N${CHAR}27) [[36., 35., 42., 0.1, 0],[36., 35., 42., 0.1, 0]]"
} >> "$TARGFILE"

echo "Appended AGN N${CHAR} (${NAME}) to $TARGFILE  [z=${ZRED}]"
