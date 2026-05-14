@echo off

setlocal

set target=ALL_BUILD
REM set target=clean

REM Building LLVM using all processors uses a lot of RAM and it can crush the
REM video driver rebooting the laptop...
set /a nproc=%NUMBER_OF_PROCESSORS% / 2
if %nproc% LSS 1 set nproc=1

pushd 3rdparty
    pushd llvm-project
        if not exist llvm\build (md llvm\build || goto :exit)
        pushd llvm\build || goto :exit
            cmake ^
                -DLLVM_ENABLE_PROJECTS=clang ^
                -DLLVM_INCLUDE_TESTS=OFF ^
                -DCMAKE_BUILD_TYPE=%build_type% .. || goto :exit
            cmake --build . --target %target% --parallel %nproc% --config %build_type% || goto :exit
            if not exist "%build_type%\bin\gcc.exe" (
                mklink /h "%build_type%\bin\gcc.exe" ^
                    "%build_type%\bin\clang.exe" || goto :exit
            )
        popd
    popd
    
    pushd tvm
        if not exist build (md build || goto :exit)
        pushd build
            copy /y ..\cmake\config.cmake .
            echo set(USE_EXAMPLE_NPU_CODEGEN ON) >> config.cmake
            echo set(USE_EXAMPLE_NPU_RUNTIME ON) >> conifg.cmake
            echo set(USE_LLVM "llvm-config --ignore-libllvm --link-static") >> config.cmake
            REM MSVC does not support this option.
            echo set(HIDE_PRIVATE_SYMBOLS OFF) >> config.cmake
            echo set(CMAKE_BUILD_TYPE %build_type%) >> config.cmake
            REM For some reason even though both LLVM are compiled with Debug
            REM this flag is needed to compile...
            echo add_compile_options("/MDd")  >> config.cmake || goto :exit
            cmake .. || goto :exit
            cmake --build . --target %target% --parallel %nproc% --config %build_type% || goto :exit
        popd
        pushd 3rdparty\tvm-ffi
            pip install --upgrade . || goto :exit
        popd

        pip install --upgrade "--target=%TVM_HOME%\python" "%TVM_HOME%\3rdparty\tvm-ffi" || goto :exit
        REM This is just needed if we want to create the wheel an install TVM in
        REM the virtual environment. Setting the PYTHONPATH allows us to modify
        REM the files in place without needing to installing it again.
        REM pip install --upgrade . || goto :exit
    popd
popd

endlocal

:exit
if %ERRORLEVEL% neq 0 echo An error occurred!
exit /b %ERRORLEVEL%