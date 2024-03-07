PWD_PATH=`pwd`
export LD_LIBRARY_PATH="$PWD_PATH/../contrib/embree/linux:${LD_LIBRARY_PATH}"
../dist/bin/Release/unittest64 --width=512 --height=512 --referencePath=../test/references/ --gtest_filter=*PerformanceTest* --gtest_output=xml:../result.xml
