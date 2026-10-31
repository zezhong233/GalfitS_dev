#!/usr/bin/env bash
# cat_profile.bash -- append a sersic profile block (P{char}) + galaxy block
#                     (G{char}) to an existing galfitS lyric file.
#
# The lyric is assumed to already contain its R/I/A sections (write those
# yourself, e.g. with a python loop). This script fills a profile with the
# default parameter set; only Pa9 (f_cont bins) and Pa10 (age bins) are
# computed from the redshift, via galfits.gsutils (needs python).
#
# Usage:
#   bash cat_profile.bash --targfile xxx.lyric --pname disk --zred 6.684 --char a
#   bash cat_profile.bash --targfile xxx.lyric --pname obj1 --zred 0.122 --char b --logm 9.5
#
# Options:
#   --targfile   existing lyric file to append to (required)
#   --pname      profile name, written to P{char}1) (required)
#   --char       single letter: profile/galaxy section index (required)
#   --zred       redshift of this galaxy (required; also used for Pa9/Pa10)
#   --logm       log10 stellar mass init (default 8.2); sets Pa14 init AND the
#                mass used by the cosmic-SFH prior that generates Pa9
#   --nosed      freeze all SED params: every vary flag from P9 on is written 0
#                (P9 f_cont, P11-P36; P10 age list has no flag; P27 stays 1)
#   --ebv        MW E(B-V) for G{char}4) (default 0)
#   --python     python interpreter with galfits installed
#                (default: $GALFITS_PYTHON or python3)

set -euo pipefail

usage() { sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; }

TARGFILE="" ; PNAME="" ; CHAR="" ; ZRED="" ; LOGM="8.2" ; NOSED=0 ; EBV="0"
PYBIN="${GALFITS_PYTHON:-python3}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --targfile) TARGFILE="$2"; shift 2 ;;
        --pname)    PNAME="$2";    shift 2 ;;
        --char)     CHAR="$2";     shift 2 ;;
        --zred)     ZRED="$2";     shift 2 ;;
        --logm)     LOGM="$2";     shift 2 ;;
        --nosed)    NOSED=1;       shift 1 ;;
        --ebv)      EBV="$2";      shift 2 ;;
        --python)   PYBIN="$2";    shift 2 ;;
        -h|--help)  usage; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

for v in TARGFILE PNAME CHAR ZRED; do
    if [[ -z "${!v}" ]]; then
        echo "ERROR: --$(tr 'A-Z' 'a-z' <<< "$v") is required" >&2; usage; exit 1
    fi
done
[[ -f "$TARGFILE" ]] || { echo "ERROR: targfile not found: $TARGFILE" >&2; exit 1; }
[[ "$CHAR" =~ ^[a-z]$ ]] || { echo "ERROR: --char must be a single lowercase letter" >&2; exit 1; }
if grep -qE "^P${CHAR}[0-9T]+\)" "$TARGFILE"; then
    echo "ERROR: profile section P${CHAR}x) already exists in $TARGFILE" >&2; exit 1
fi
if grep -qE "^G${CHAR}[0-9]+\)" "$TARGFILE"; then
    echo "ERROR: galaxy section G${CHAR}x) already exists in $TARGFILE" >&2; exit 1
fi

# vary flag for all SED params from P9 on (0 if --nosed); P27 is not a fit
# parameter (use_sed switch) and always stays 1
VF=$(( 1 - NOSED ))

# ---- compute Pa9 (f_cont bins, cosmic-SFH prior) and Pa10 (age bins) ----
PYOUT=$("$PYBIN" - "$ZRED" "$LOGM" "$VF" <<'PYEOF' 2>/dev/null
import sys
z, logm, vf = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
from galfits.gsutils import z_to_agebin, _generate_cosmic_sfh_prior
ages = [float(a) for a in z_to_agebin(z)]
fcont = [float(v) for v in _generate_cosmic_sfh_prior(z, ages, logm)]
print('[' + ', '.join('[{0}, -5, 0, 0.1, {1}]'.format(repr(v), vf) for v in fcont) + ']')
print('[' + ', '.join(repr(a) for a in ages) + ']')
PYEOF
) || { echo "ERROR: python failed to compute Pa9/Pa10. Is galfits installed for '$PYBIN'?" >&2
       echo "       Pass --python /path/to/python or set GALFITS_PYTHON." >&2; exit 1; }

mapfile -t PLINES <<< "$(grep '^\[' <<< "$PYOUT")"
if [[ ${#PLINES[@]} -ne 2 ]]; then
    echo "ERROR: unexpected python output for Pa9/Pa10:" >&2
    echo "$PYOUT" >&2
    exit 1
fi
PA9="${PLINES[0]}"
PA10="${PLINES[1]}"

# ---- append profile + galaxy blocks ----
{
    echo ""
    echo "# Profile ${CHAR}: ${PNAME} (sersic, z=${ZRED})"
    echo "P${CHAR}1) ${PNAME}"
    echo "P${CHAR}2) sersic"
    echo "P${CHAR}3) [0.00, -0.40, 0.40, 0.02, 1]"
    echo "P${CHAR}4) [0.00, -0.40, 0.40, 0.02, 1]"
    echo "P${CHAR}5) [0.03 ,0.012, 1.02, 0.01, 1]"
    echo "P${CHAR}6) [2.5, 0.2, 8, 0.2, 1]"
    echo "P${CHAR}7) [0, -180, 180, 5, 1]"
    echo "P${CHAR}8) [0.8, 0.1, 1, 0.05, 1]"
    echo "P${CHAR}9) ${PA9}"
    echo "P${CHAR}10) ${PA10}"
    echo "P${CHAR}11) [[0.004, 0.0001, 0.04, 0.001, ${VF}]]"
    echo "P${CHAR}12) [[0., 0, 5.1, 0.1, 0]]"
    echo "P${CHAR}13) [0, 40, 500, 1, 0]"
    echo "P${CHAR}14) [${LOGM}, 6, 12, 0.1, ${VF}]"
    echo "P${CHAR}15) bins"
    echo "P${CHAR}16) [-1, -4, -2, 0.1, 0]"
    echo "P${CHAR}26) [0, 0, 5, 0.1, 0]"
    echo "P${CHAR}27) 1"
    echo "P${CHAR}28) [6, 4.5, 10, 0.1, 0]"
    echo "P${CHAR}29) [1.0, 0.1, 50, 10, 0]"
    echo "P${CHAR}30) [2, 0.45, 7, 2, 0]"
    echo "P${CHAR}31) [1.0, 1.0, 3.0, 0.1, 0]"
    echo "P${CHAR}32) [0.01, 0, 1.0, 0.05, 0]"
    echo "P${CHAR}33) [[0.004, 0.0001, 0.04, 0.001, ${VF}]]"
    echo "P${CHAR}34) [0.3, 0, 4, 1, ${VF}]"
    echo "P${CHAR}35) [1, 0, 2, 0.3, ${VF}]"
    echo "P${CHAR}36) [-0.7, -1, 0.4, 0.1, ${VF}]"
    echo ""
    echo "# Galaxy ${CHAR}"
    echo "G${CHAR}1) ${CHAR}"
    echo "G${CHAR}2) ['${CHAR}']"
    echo "G${CHAR}3) [${ZRED}, 0.01, 12, 0.01, 0]"
    echo "G${CHAR}4) ${EBV}"
    echo "G${CHAR}5) [1., 0.8, 1.2, 0.05, 0]"
    echo "G${CHAR}6) []"
    echo "G${CHAR}7) 1"
} >> "$TARGFILE"

echo "Appended profile P${CHAR} (${PNAME}) + galaxy G${CHAR} to $TARGFILE  [z=${ZRED}, logM=${LOGM}, nosed=${NOSED}]"
