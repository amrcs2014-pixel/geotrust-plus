#!/bin/bash
cd ~/ns-3-geotrust
mkdir -p scratch_disabled
mv scratch/rangeguard.cc scratch/rangeguard_ids.cc scratch_disabled/ 2>/dev/null
./ns3 build geotrust_attack > ~/geo_build2.log 2>&1
echo BUILD_EXIT=$? >> ~/geo_build2.log
tail -5 ~/geo_build2.log
