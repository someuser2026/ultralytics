#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: bash jobs/train/hpc/submit_resume_training.sh <full|2hr> <last.pt> [last.pt ...]"
    echo "Optional environment overrides: DRY_RUN=1 DEVICE=0 BATCH=<batch> WANDB=true"
}

if [[ "$#" -lt 2 ]]; then
    usage >&2
    exit 2
fi

profile="$1"
shift

case "${profile}" in
    full)
        pbs_script="jobs/train/hpc/planet_full.pbs"
        ;;
    2hr)
        pbs_script="jobs/train/hpc/planet_full_2hr_walltime.pbs"
        ;;
    *)
        echo "Error: Invalid PBS profile '${profile}'. Expected 'full' or '2hr'." >&2
        usage >&2
        exit 2
        ;;
esac

if [[ ! -f "${pbs_script}" ]]; then
    echo "Error: PBS wrapper not found: ${pbs_script}" >&2
    exit 1
fi

checkpoints=("$@")
for checkpoint in "${checkpoints[@]}"; do
    if [[ "${checkpoint}" == *,* ]]; then
        echo "Error: Checkpoint paths cannot contain commas because qsub -v uses comma separators: ${checkpoint}" >&2
        exit 2
    fi
    if [[ ! -f "${checkpoint}" ]]; then
        echo "Error: Checkpoint not found: ${checkpoint}" >&2
        exit 1
    fi
    if [[ "$(basename "${checkpoint}")" != "last.pt" ]]; then
        echo "Error: Only resumable last.pt checkpoints are accepted: ${checkpoint}" >&2
        exit 2
    fi
done

dry_run="${DRY_RUN:-0}"
device="${DEVICE:-}"
batch="${BATCH:-}"
wandb="${WANDB:-true}"

for checkpoint in "${checkpoints[@]}"; do
    run_dir="$(basename "$(dirname "$(dirname "${checkpoint}")")")"
    safe_run="$(printf '%s' "${run_dir}" | tr -cs '[:alnum:]_-' '_')"
    job_name="resume_${safe_run}"
    varlist="RESUME=true,CHECKPOINT=${checkpoint},WANDB=${wandb}"
    [[ -n "${device}" ]] && varlist+=",DEVICE=${device}"
    [[ -n "${batch}" ]] && varlist+=",BATCH=${batch}"

    qsub_args=(-V -v "${varlist}" -N "${job_name}" "${pbs_script}")
    if [[ "${dry_run}" == "1" ]]; then
        printf 'DRY_RUN qsub'
        for arg in "${qsub_args[@]}"; do
            printf ' %q' "${arg}"
        done
        printf '\n'
    else
        echo "Submitting ${checkpoint} with profile=${profile} as ${job_name}"
        qsub "${qsub_args[@]}"
    fi
done
