mkdir -p build
cd build
conan install ../conan --output-folder=. --build missing
cmake ../src -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release
cmake --build .
