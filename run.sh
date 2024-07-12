#!/usr/bin/env bash

# bash safe mode. look at `set --help` to see what these are doing
set -euxo pipefail 

cd $(dirname $0)
exec dist/main $@
source .env
./setup.sh
source .venv/bin/activate
# # Be sure to use `exec` so that termination signals reach the python process,
# # or handle forwarding termination signals manually
echo which python3
python3 -m src.main $@
