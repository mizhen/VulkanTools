echo off
REM
REM This batch file builds both 32-bit and 64-bit versions of the loader.
REM It is assumed that the developer has run the update_external_sources.bat
REM file prior to running this.
REM

REM Determine the appropriate CMake strings for the current version of Visual Studio
echo Determining VS version
python .\determine_vs_version.py > vsversion.tmp
set /p VS_VERSION=< vsversion.tmp
echo Detected Visual Studio Version as %VS_VERSION%

REM Cleanup the file we used to collect the VS version output since it's no longer needed.
del /Q /F vsversion.tmp

rmdir /Q /S build
rmdir /Q /S build32

REM *******************************************
REM 64-bit LoaderAndTools build
REM *******************************************
mkdir build
pushd build

echo Generating 64-bit spirv-tools CMake files for Visual Studio %VS_VERSION%
cmake -G "Visual Studio %VS_VERSION% Win64" ..
   
echo Building 64-bit Debug LoaderAndTools 
msbuild ALL_BUILD.vcxproj /p:Platform=x64 /p:Configuration=Debug /verbosity:quiet
   
REM Check for existence of one DLL, even though we should check for all results
if not exist .\loader\Debug\vulkan-1.dll (
   echo.
   echo LoaderAndTools 64-bit Debug build failed!
   set errorCode=1
)
   
echo Building 64-bit Release LoaderAndTools 
msbuild ALL_BUILD.vcxproj /p:Platform=x64 /p:Configuration=Release /verbosity:quiet

REM Check for existence of one DLL, even though we should check for all results
if not exist .\loader\Release\vulkan-1.dll (
   echo.
   echo LoaderAndTools 64-bit Release build failed!
   set errorCode=1
)

popd
 
REM *******************************************
REM 32-bit LoaderAndTools build
REM *******************************************
mkdir build32
pushd build32
  
echo Generating 32-bit LoaderAndTools CMake files for Visual Studio %VS_VERSION%
cmake -G "Visual Studio %VS_VERSION%" ..
   
echo Building 32-bit Debug LoaderAndTools 
msbuild ALL_BUILD.vcxproj /p:Platform=x86 /p:Configuration=Debug /verbosity:quiet
   
REM Check for existence of one DLL, even though we should check for all results
if not exist .\loader\Debug\vulkan-1.dll (
   echo.
   echo LoaderAndTools 32-bit Debug build failed!
   set errorCode=1
)
   
echo Building 32-bit Release LoaderAndTools 
msbuild ALL_BUILD.vcxproj /p:Platform=x86 /p:Configuration=Release /verbosity:quiet

REM Check for existence of one DLL, even though we should check for all results
if not exist .\loader\Release\vulkan-1.dll (
   echo.
   echo LoaderAndTools 32-bit Release build failed!
   set errorCode=1
)

popd

