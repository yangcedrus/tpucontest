#!/bin/bash
CURRENT_DIR=$(cd $(dirname "${BASH_SOURCE[0]}") && pwd)
mkdir -p $CURRENT_DIR/../toolchains_dir
pushd $CURRENT_DIR/../toolchains_dir >>/dev/null
linaro_toolchain="gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu"
if [ ! -d "$linaro_toolchain" ]; then
    if [ ! -e "${linaro_toolchain}.tar.xz" ]; then
        wget https://releases.linaro.org/components/toolchain/binaries/6.3-2017.05/aarch64-linux-gnu/${linaro_toolchain}.tar.xz --no-check-certificate
    fi
    tar xvf "${linaro_toolchain}.tar.xz"
fi

arm_toolchain="gcc-arm-none-eabi-7-2017-q4-major"
if [ ! -d "${arm_toolchain}" ]; then
    if [ ! -e "${arm_toolchain}.tar.bz2" ]; then
        wget https://developer.arm.com/-/media/Files/downloads/gnu-rm/7-2017q4/${arm_toolchain}-linux.tar.bz2 --no-check-certificate
    fi
    tar xvf "${arm_toolchain}-linux.tar.bz2"
fi

export CROSS_TOOLCHAINS=`pwd`
echo "export CROSS_TOOLCHAINS=$CROSS_TOOLCHAINS"
popd >>/dev/null
