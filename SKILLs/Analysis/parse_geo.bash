#!/usr/bin/env bash
# parse_gsgeo.bash -- extract geometry parameters from a galfitS .gssummary
#                     into a small summary file for the next fitting stage.
#
# Extracted parameters:
#   per profile : P3-P8  -> {p}_xcen {p}_ycen {p}_Re {p}_n {p}_ang {p}_axrat
#   per AGN (NP): N4-N5  -> xcen_{n} ycen_{n}
#
# Usage:
#   bash parse_gsgeo.bash --targfile xxx.gssummary --dst out_dir --name xxx_geoinit.gssummary
#
# Then feed to the image-SED stage with:
#   galfitS.py stage2.lyric --workplace ... --readsummary out_dir/xxx_geoinit.gssummary
#
# NOTE: --readsummary tightens bounds to best +/- 10*brute_step of the NEW
# lyric. Check the source summary for boundary hits before using this file.

set -euo pipefail

usage() {
    sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'
}

TARGFILE=""
DST="."
NAME=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --targfile) TARGFILE="$2"; shift 2 ;;
        --dst)      DST="$2";      shift 2 ;;
        --name)     NAME="$2";     shift 2 ;;
        -h|--help)  usage; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ -z "$TARGFILE" ]]; then
    echo "ERROR: --targfile is required" >&2
    usage
    exit 1
fi
if [[ ! -f "$TARGFILE" ]]; then
    echo "ERROR: targfile not found: $TARGFILE" >&2
    exit 1
fi
if [[ -z "$NAME" ]]; then
    base=$(basename "$TARGFILE")
    NAME="${base%.gssummary}_geoinit.gssummary"
fi

mkdir -p "$DST"
OUTPATH="$DST/$NAME"

awk -v targ="$TARGFILE" -v out="$OUTPATH" '
BEGIN {
    np = split("xcen ycen Re n ang axrat", PK, " ")  # P3..P8, output order
    na = split("xcen ycen", AK, " ")                 # N4, N5
}
!/^#/ && NF >= 2 && $1 != "pname" {
    name = $1; val = $2
    matched = 0
    for (i = 1; i <= np; i++) {                      # {p}_{key} profiles
        suf = "_" PK[i]
        if (length(name) > length(suf) &&
            substr(name, length(name) - length(suf) + 1) == suf) {
            prof = substr(name, 1, length(name) - length(suf))
            if (!(prof in pseen)) { pseen[prof] = 1; porder[++nprof] = prof }
            cell[prof, PK[i]] = name; value[name] = val
            matched = 1
            break
        }
    }
    if (matched) next
    for (i = 1; i <= na; i++) {                      # {key}_{n} AGN centers
        pre = AK[i] "_"
        if (substr(name, 1, length(pre)) == pre) {
            agn = substr(name, length(pre) + 1)
            if (!(agn in aseen)) { aseen[agn] = 1; aorder[++nagn] = agn }
            cell[agn, AK[i]] = name; value[name] = val
            break
        }
    }
}
END {
    if (nprof == 0 && nagn == 0) {
        print "ERROR: no geometry parameters found in " targ > "/dev/stderr"
        exit 1
    }
    print "# geometry-only summary, extracted from: " targ                     > out
    print "# content: per-profile P3-P8 (xcen ycen Re n ang axrat); AGN N4-N5" > out
    print "# usage: galfitS.py <stage2.lyric> --readsummary " out              > out
    print "pname best_value"                                                   > out
    n = 0
    for (i = 1; i <= nprof; i++) {
        prof = porder[i]
        for (j = 1; j <= np; j++) {
            k = PK[j]
            if ((prof, k) in cell) {
                nm = cell[prof, k]
                print nm, value[nm] > out
                n++
            } else {
                print "WARNING: " prof "_" k " not found in source summary" > "/dev/stderr"
            }
        }
    }
    for (i = 1; i <= nagn; i++) {
        agn = aorder[i]
        for (j = 1; j <= na; j++) {
            k = AK[j]
            if ((agn, k) in cell) {
                nm = cell[agn, k]
                print nm, value[nm] > out
                n++
            }
        }
    }
    printf "Wrote %d geometry parameters (%d profile(s), %d AGN) to %s\n", n, nprof, nagn, out
}
' "$TARGFILE"
