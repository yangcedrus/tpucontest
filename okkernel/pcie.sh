#!/bin/bash
make clean
make okk
cd build/pcie
./load_firmware
cd -
