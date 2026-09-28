#!/bin/bash
# GeoTrust+ ns-3 build (copy of the user's ns-3-dev with RangeGuard patches + GeoTrust+ MprMode 2)
cd ~/ns-3-geotrust
sed -i 's/                          MakeUintegerChecker<uint32_t>(0, 1))/                          MakeUintegerChecker<uint32_t>(0, 2))/' src/olsr/model/olsr-routing-protocol.cc
grep -c 'uint32_t>(0, 2)' src/olsr/model/olsr-routing-protocol.cc
./ns3 configure -d optimized --enable-modules "olsr;wifi;mobility;applications;flow-monitor;internet" \
    --disable-tests --disable-examples > ~/geo_configure.log 2>&1
echo CONFIGURE_EXIT=$? >> ~/geo_configure.log
./ns3 build > ~/geo_build.log 2>&1
echo BUILD_EXIT=$? >> ~/geo_build.log
