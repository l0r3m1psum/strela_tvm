@echo off

setlocal

set target=ALL_BUILD
REM set target=clean
set build_type=Debug

pushd 3rdparty
    pushd llvm-project
        if not exist llvm\build (md llvm\build || goto :exit)
        pushd llvm\build || goto :exit
            cmake ^
                -DLLVM_ENABLE_PROJECTS=clang ^
                -DLLVM_INCLUDE_TESTS=OFF ^
                -DCMAKE_BUILD_TYPE=%build_type% .. || goto :exit
            cmake --build . --target %target% --parallel %NUMBER_OF_PROCESSORS% --config %build_type% || goto :exit
            REM if not exist "%installdir%\Programs\LLVM\bin\gcc.exe" (
            REM     mklink /h "%installdir%\Programs\LLVM\bin\gcc.exe" ^
            REM         "%installdir%\Programs\LLVM\bin\clang.exe" || goto :exit
            REM )
        popd
    popd
    
    pushd tvm
        if not exist build (md build || goto :exit)
        pushd build
            copy /y ..\cmake\config.cmake .
            echo set(USE_EXAMPLE_NPU_CODEGEN ON) >> config.cmake
            echo set(USE_EXAMPLE_NPU_RUNTIME ON) >> conifg.cmake
            echo set(USE_LLVM "%CD:\=/%/../../llvm-project/llvm/build/%build_type%/bin/llvm-config --ignore-libllvm --link-static") >> config.cmake
            echo set(HIDE_PRIVATE_SYMBOLS ON) >> config.cmake
            echo set(CMAKE_BUILD_TYPE %build_type%) >> config.cmake
            echo add_compile_options("/MDd")  >> config.cmake || goto :exit
            cmake .. || goto :exit
            cmake --build . --target %target% --parallel %NUMBER_OF_PROCESSORS% --config %build_type% || goto :exit
        popd
        pushd 3rdparty\tvm-ffi
            pip install . || goto :exit
        popd
        set "TVM_HOME=%CD%"
        pip install "--target=%TVM_HOME%\python% %%TVM_HOME%\3rdparty\tvm-ffi% || goto :exit
        set "PYTHONPATH=%TVM_HOME%/python:%PYTHONPATH%"
        pip install . || goto :exit
    popd
popd

endlocal