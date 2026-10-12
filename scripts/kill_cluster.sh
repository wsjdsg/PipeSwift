#!/usr/bin/env bash

set -euo pipefail

if [[ $# -gt 0 ]]; then
  HOSTS=("$@")
elif [[ -n "${PIPESWIFT_HOSTS:-}" ]]; then
  IFS=',' read -r -a HOSTS <<< "${PIPESWIFT_HOSTS}"
else
  echo "usage: $0 <host> [<host> ...]" >&2
  echo "   or: PIPESWIFT_HOSTS=host1,host2 $0" >&2
  exit 2
fi

for host in "${HOSTS[@]}"; do
  echo "Stopping PipeSwift on ${host}"
  pdsh -w "ssh:${host}" \
    'pkill -f "[p]ython.*pipeswift" || true; lsof -t -i:3678 | xargs -r kill -9'
done
