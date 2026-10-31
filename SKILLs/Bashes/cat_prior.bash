#!/usr/bin/env bash
# cat_prior.bash -- append prior entries to a galfitS .prior file.
#
# Two prior types (parsed in gsutils.read_config_file):
#   SFH_T : Student-T prior on adjacent-bin SFH ratios for one profile.
#           Appends a new section with the next free label letter:
#             SFH_T{x}1) {pname}
#             SFH_T{x}2) {cosmic|flat}
#   GP    : Gaussian priors. Single line `GP) [names...]`; mu/sigma are taken
#           from the lyric init/brute_step. New names are MERGED into the
#           existing GP) line (created if absent).
#
# Short param names are mapped to full galfitS names via the two naming
# conventions (see SKILLs/Basics/NAME_FORMAT.md):
#   suffix style {p}_{key} : xcen ycen Re n ang axrat dust2 dust_ratio ...
#   prefix style {key}_{p} : logM logU AVbump logMdust Umin qPAH alphad gammad
#
# Usage:
#   bash cat_prior.bash --targfile xxx.prior --type SFH_T --pname disk --type_sfh cosmic
#   bash cat_prior.bash --targfile xxx.prior --type GP --pname com1 \
#                        --params '["dust2","dust_ratio","qPAH"]'
#
# Options:
#   --targfile   prior file to write (created if missing) (required)
#   --type       SFH_T | GP (required)
#   --pname      profile name the prior applies to (required for both types;
#                SFH_T{x}1) value / GP name construction; --name also accepted)
#   --type_sfh   cosmic | flat (SFH_T only, required)
#   --params     short param names, python-list or comma-separated (GP only)

set -euo pipefail

usage() { sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; }

TARGFILE="" ; TYPE="" ; PNAME="" ; TYPE_SFH="" ; PARAMS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --targfile) TARGFILE="$2";  shift 2 ;;
        --type)     TYPE="$2";      shift 2 ;;
        --pname|--name) PNAME="$2"; shift 2 ;;
        --type_sfh) TYPE_SFH="$2";  shift 2 ;;
        --params)   PARAMS="$2";    shift 2 ;;
        -h|--help)  usage; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

[[ -n "$TARGFILE" && -n "$TYPE" ]] || { echo "ERROR: --targfile and --type are required" >&2; usage; exit 1; }
TYPE=$(tr 'a-z' 'A-Z' <<< "$TYPE")
[[ "$TYPE" == "SFH_T" || "$TYPE" == "GP" ]] || { echo "ERROR: --type must be SFH_T or GP" >&2; exit 1; }

touch "$TARGFILE"

# short-name -> naming style: pre = {key}_{pname}, suf = {pname}_{key}
declare -A PARAM_STYLE=(
    [logM]=pre [logU]=pre [AVbump]=pre [logMdust]=pre
    [Umin]=pre [qPAH]=pre [alphad]=pre [gammad]=pre
    [xcen]=suf [ycen]=suf [Re]=suf [n]=suf [ang]=suf [axrat]=suf
    [f_cont_value]=suf [f_cont_in]=suf [f_cont_out]=suf
    [age_value]=suf [age_in]=suf [age_out]=suf
    [Z_value]=suf [Z_in]=suf [Z_out]=suf
    [Av_value]=suf [Av_in]=suf [Av_out]=suf
    [Zneb_value]=suf [Zneb_in]=suf [Zneb_out]=suf
    [sigma]=suf [dust2]=suf [dust_ratio]=suf [dust_index]=suf
)

if [[ "$TYPE" == "SFH_T" ]]; then
    [[ -n "$PNAME" && -n "$TYPE_SFH" ]] || { echo "ERROR: SFH_T needs --pname and --type_sfh" >&2; exit 1; }
    [[ "$TYPE_SFH" == "cosmic" || "$TYPE_SFH" == "flat" ]] || { echo "ERROR: --type_sfh must be cosmic or flat" >&2; exit 1; }
    if grep -qE "^SFH_T[a-z]1\) +${PNAME} *$" "$TARGFILE"; then
        echo "ERROR: SFH_T prior for '${PNAME}' already exists in $TARGFILE" >&2; exit 1
    fi
    # next free label letter (a, b, ...)
    LAST=$(grep -oE '^SFH_T[a-z]' "$TARGFILE" | tail -1 | grep -oE '[a-z]$' || true)
    if [[ -z "$LAST" ]]; then
        LABEL=a
    else
        LABEL=$(printf "\\$(printf '%03o' $(( $(printf '%d' "'$LAST") + 1 )))")
    fi
    [[ "$LABEL" =~ ^[a-z]$ ]] || { echo "ERROR: ran out of section labels (past z)" >&2; exit 1; }
    [[ -s "$TARGFILE" ]] && echo "" >> "$TARGFILE"
    {
        echo "SFH_T${LABEL}1) ${PNAME}"
        echo "SFH_T${LABEL}2) ${TYPE_SFH}"
    } >> "$TARGFILE"
    echo "Appended SFH_T${LABEL} prior: ${PNAME} (${TYPE_SFH}) -> $TARGFILE"

else  # GP
    [[ -n "$PNAME" && -n "$PARAMS" ]] || { echo "ERROR: GP needs --pname and --params" >&2; exit 1; }

    # parse short names: strip [] ' " and spaces, split on commas
    mapfile -t SHORTS <<< "$(sed 's/[][]//g; s/["'"'"']//g; s/ //g' <<< "$PARAMS" | tr ',' '\n' | sed '/^$/d')"
    [[ ${#SHORTS[@]} -gt 0 ]] || { echo "ERROR: --params parsed to an empty list" >&2; exit 1; }

    # map to full galfitS names
    NEW=()
    for s in "${SHORTS[@]}"; do
        style="${PARAM_STYLE[$s]:-}"
        if [[ -z "$style" ]]; then
            echo "ERROR: unknown param short-name '$s'. Valid keys:" >&2
            printf '  %s\n' "${!PARAM_STYLE[@]}" | sort >&2
            exit 1
        fi
        if [[ "$style" == "pre" ]]; then NEW+=("${s}_${PNAME}"); else NEW+=("${PNAME}_${s}"); fi
    done

    # existing GP) names (if any)
    OLD=()
    GP_EXISTS=0
    if grep -qE '^GP\)' "$TARGFILE"; then
        GP_EXISTS=1
        mapfile -t OLD <<< "$(grep -E '^GP\)' "$TARGFILE" | head -1 \
            | sed 's/^GP) *//' | sed 's/[][]//g; s/["'"'"']//g; s/ //g' | tr ',' '\n' | sed '/^$/d')"
    fi

    # merge, dedup preserving order
    MERGED=("${OLD[@]}")
    for n in "${NEW[@]}"; do
        dup=0
        for o in ${MERGED[@]+"${MERGED[@]}"}; do [[ "$o" == "$n" ]] && { dup=1; break; }; done
        [[ $dup -eq 0 ]] && MERGED+=("$n")
    done

    # build the GP) line
    GPLINE="GP) ["
    for i in "${!MERGED[@]}"; do
        [[ $i -gt 0 ]] && GPLINE+=", "
        GPLINE+="\"${MERGED[$i]}\""
    done
    GPLINE+="]"

    if [[ $GP_EXISTS -eq 1 ]]; then
        # replace existing GP) line in place
        awk -v gp="$GPLINE" '{if ($0 ~ /^GP\)/) print gp; else print $0}' "$TARGFILE" > "$TARGFILE.tmp"
        mv "$TARGFILE.tmp" "$TARGFILE"
        echo "Updated GP) line in $TARGFILE (+$(( ${#MERGED[@]} - ${#OLD[@]} )) new):"
    else
        [[ -s "$TARGFILE" ]] && echo "" >> "$TARGFILE"
        echo "$GPLINE" >> "$TARGFILE"
        echo "Appended GP) line to $TARGFILE:"
    fi
    echo "  $GPLINE"
fi
