@echo off
rem synesthesia-engine build (MinGW g++)
rem -static: avoid clashing with another libstdc++-6.dll on PATH (e.g. the one shipped with Git)
cd /d "%~dp0"
set FLAGS=-O2 -std=c++20 -Wall -Wextra -static
echo [1/2] composer.exe ...
g++ %FLAGS% composer.cpp -o composer.exe
if errorlevel 1 exit /b 1
echo [2/2] test_fusion.exe ...
g++ %FLAGS% test_fusion.cpp -o test_fusion.exe
if errorlevel 1 exit /b 1
echo OK
