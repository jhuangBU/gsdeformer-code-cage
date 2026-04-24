#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

# make GMorpho
cd GMorpho
make debug 
cd ..

# make Broxy
make debug 
