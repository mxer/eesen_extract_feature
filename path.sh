#!/usr/bin/env bash
#export EESEN_ROOT=`pwd`/../../..
export EESEN_ROOT=`pwd`
export PATH=$PWD/utils/:${EESEN_ROOT}/src/netbin:${EESEN_ROOT}/src/featbin:${EESEN_ROOT}/src/decoderbin:${EESEN_ROOT}/src/fstbin:${EESEN_ROOT}/tools/openfst/bin:${EESEN_ROOT}/tools/irstlm/bin/:$PWD:$PATH
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${EESEN_ROOT}/tools/openfst/lib
export LC_ALL=C

if [[ ! -z ${acwt+x} ]]; then
    # let's assume we're decoding
    export PATH=$EESEN_ROOT/src-nogpu/netbin:$PATH
    echo "Preferring non-gpu netbin code"
fi
