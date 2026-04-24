#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

# make GMorpho
cd GMorpho
qmake CONFIG+=debug_and_release
# note: extra sed used to remove : \ entries in Makefile
sed -i 's/\\: \\/\/ \\/g' Makefile 
sed -i 's/\\: \\/\/ \\/g' Makefile.Release
sed -i 's/\\: \\/\/ \\/g' Makefile.Debug
make debug 
cd ..

# make Broxy
qmake CONFIG+=debug_and_release
make debug 
