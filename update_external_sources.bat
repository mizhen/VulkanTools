@echo off
REM Update source for glslang

REM Determine the appropriate CMake strings for the current version of Visual Studio
echo Determining VS version
python .\scripts\determine_vs_version.py > vsversion.tmp
set /p VS_VERSION=< vsversion.tmp
echo Detected Visual Studio Version as %VS_VERSION%

REM Cleanup the file we used to collect the VS version output since it's no longer needed.
del /Q /F vsversion.tmp

setlocal EnableDelayedExpansion
set errorCode=0
set BUILD_DIR=%~dp0
set BASE_DIR="%BUILD_DIR%external"
set REVISION_DIR="%BUILD_DIR%external_revisions"
set GLSLANG_DIR=%BASE_DIR%\glslang
set JSONCPP_DIR=%BASE_DIR%\jsoncpp

REM // ======== Parameter parsing ======== //

   set arg-use-implicit-component-list=1
   set arg-do-glslang=0
   set arg-do-jsoncpp=0
   set arg-no-sync=0
   set arg-no-build=0

   :parameterLoop

      if "%1"=="" goto:parameterContinue

      if "%1" == "--glslang" (
         set arg-do-glslang=1
         set arg-use-implicit-component-list=0
         echo Building glslang ^(%1^)
         shift
         goto:parameterLoop
      )

      if "%1" == "-g" (
         set arg-do-glslang=1
         set arg-use-implicit-component-list=0
         echo Building glslang ^(%1^)
         shift
         goto:parameterLoop
      )

      if "%1" == "--no-sync" (
         set arg-no-sync=1
         echo Skipping sync ^(%1^)
         shift
         goto:parameterLoop
      )

      if "%1" == "--no-build" (
         set arg-no-build=1
         echo Skipping build ^(%1^)
         shift
         goto:parameterLoop
      )
      if "%1" == "--jsoncpp" (
         set arg-do-jsoncpp=1
         set arg-use-implicit-component-list=0
         echo Building jsoncpp ^(%1^)
         shift
         goto:parameterLoop
      )

      if "%1" == "-j" (
         set arg-do-jsoncpp=1
         set arg-use-implicit-component-list=0
         echo Building jsoncpp ^(%1^)
         shift
         goto:parameterLoop
      )

      if "%1" == "--all" (
         set arg-do-glslang=1
         set arg-do-jsoncpp=1
         set arg-use-implicit-component-list=0
         echo Building glslang, spirv-tools ^(%1^)
         shift
         goto:parameterLoop
      )

      if "%1" == "--spirv-tools" (
         echo --spirv-tools argument has been deprecated and is no longer necessary
         shift
         goto:parameterLoop
      )

      if "%1" == "-s" (
         echo --s argument has been deprecated and is no longer necessary
         shift
         goto:parameterLoop
      )

      if "%1" == "--all" (
         echo --all argument has been deprecated and is no longer necessary
         set arg-do-glslang=1
         set arg-use-implicit-component-list=0
         echo Building glslang ^(%1^)
         shift
         goto:parameterLoop
      )

      echo.
      echo Unrecognized option "%1"
      echo.
      echo usage: update_external_sources.bat [options]
      echo.
      echo   Available options:
      echo     -g ^| --glslang      enable glslang component
      echo     -j ^| --jsoncpp      enable jsoncpp component
      echo     --all               enable all components
      echo     --no-sync           skip sync from git
      echo     --no-build          skip build
      echo.
      echo   If any component enables are provided, only those components are enabled.
      echo   If no component enables are provided, all components are enabled.
      echo.
      echo   Sync uses git to pull a specific revision.
      echo   Build configures CMake, builds Release and Debug.


      goto:error

   :parameterContinue

   if %arg-use-implicit-component-list% equ 1 (
      echo Building glslang
      set arg-do-glslang=1
      set arg-do-jsoncpp=1
   )

   set sync-glslang=0
   set sync-jsoncpp=0
   set build-glslang=0
   set build-jsoncpp=0
   set check-glslang-build-dependencies=0

   if %arg-do-glslang% equ 1 (
      if %arg-no-sync% equ 0 (
         set sync-glslang=1
      )
      if %arg-no-build% equ 0 (
         set check-glslang-build-dependencies=1
         set build-glslang=1
      )
   )

   if %arg-do-jsoncpp% equ 1 (
      if %arg-no-sync% equ 0 (
         set sync-jsoncpp=1
      )
      if %arg-no-build% equ 0 (
         set build-jsoncpp=1
      )
   )

   REM this is a debugging aid that can be enabled while debugging command-line parsing
   if 0 equ 1 (
      set arg
      set sync-glslang
      set build-glslang
      set check-glslang-build-dependencies
      goto:error
   )

REM // ======== end Parameter parsing ======== //


REM // ======== Dependency checking ======== //
   REM git is required for all paths
   for %%X in (git.exe) do (set FOUND=%%~$PATH:X)
   if not defined FOUND (
      echo Dependency check failed:
      echo   git.exe not found
      echo   Git for Windows can be downloaded here:  https://git-scm.com/download/win
      echo   Install and ensure git.exe makes it into your PATH
      set errorCode=1
   )

   if %check-glslang-build-dependencies% equ 1 (
      for %%X in (cmake.exe) do (set FOUND=%%~$PATH:X)
      if not defined FOUND (
         echo Dependency check failed:
         echo   cmake.exe not found
         echo   Get CMake for Windows here:  http://www.cmake.org/cmake/resources/software.html
         echo   Install and ensure each makes it into your PATH, default is "C:\Program Files (x86)\CMake\bin"
         set errorCode=1
      )
   )


   REM goto:main

REM // ======== end Dependency checking ======== //

:main

if %errorCode% neq 0 (goto:error)

REM Read the target versions from external file, which is shared with Linux script

if not exist %REVISION_DIR%\glslang_giturl (
   echo.
   echo Missing glslang_giturl file!  Place it in %REVISION_DIR% with git repo URL in it.
   set errorCode=1
   goto:error
)

if not exist %REVISION_DIR%\glslang_revision (
   echo.
   echo Missing glslang_revision file!  Place it in %REVISION_DIR% with target version in it.
   set errorCode=1
   goto:error
)


if not exist %REVISION_DIR%\jsoncpp_revision (
   echo.
   echo Missing jsoncpp_revision file!  Place it in %REVISION_DIR% with target version in it.
   set errorCode=1
   goto:error
)

set /p GLSLANG_GITURL= < %REVISION_DIR%\glslang_giturl
set /p GLSLANG_REVISION= < %REVISION_DIR%\glslang_revision
set /p JSONCPP_REVISION= < %REVISION_DIR%\jsoncpp_revision

echo GLSLANG_GITURL=%GLSLANG_GITURL%
echo GLSLANG_REVISION=%GLSLANG_REVISION%
echo JSONCPP_REVISION=%JSONCPP_REVISION%


echo Creating and/or updating glslang in %BASE_DIR%

if %sync-glslang% equ 1 (
   if not exist %GLSLANG_DIR% (
      call:create_glslang
   )
   if %errorCode% neq 0 (goto:error)
   call:update_glslang
   if %errorCode% neq 0 (goto:error)
)


if %sync-jsoncpp% equ 1 (
   if exist %JSONCPP_DIR% (
      rd /S /Q %JSONCPP_DIR%
   )
   if %errorlevel% neq 0 (goto:error)
   if not exist %JSONCPP_DIR% (
      call:create_jsoncpp
   )
   if %errorCode% neq 0 (goto:error)
   call:update_jsoncpp
   if %errorCode% neq 0 (goto:error)
)

if %build-glslang% equ 1 (
   call:build_glslang
   if %errorCode% neq 0 (goto:error)
)


if %build-jsoncpp% equ 1 (
   call:build_jsoncpp
   if %errorCode% neq 0 (goto:error)
)

echo.
echo Exiting
goto:finish

:error
echo.
echo Halting due to error
set errorCode=1
goto:finish

:finish
if not "%cd%\" == "%BUILD_DIR%" ( cd %BUILD_DIR% )
exit /b %errorCode%


REM // ======== Functions ======== //

:create_glslang
   echo.
   echo Creating local glslang repository %GLSLANG_DIR%)
   mkdir %GLSLANG_DIR%
   cd %GLSLANG_DIR%
   git clone %GLSLANG_GITURL% .
   git checkout %GLSLANG_REVISION%
   python.exe .\update_glslang_sources.py
   if not exist %GLSLANG_DIR%\SPIRV (
       echo glslang source download failed!
       set errorCode=1
   )
goto:eof

:update_glslang
   echo.
   echo Updating %GLSLANG_DIR%
   cd %GLSLANG_DIR%
   git fetch --all
   git checkout %GLSLANG_REVISION%
   python.exe .\update_glslang_sources.py
goto:eof

:build_glslang
   echo.
   echo Building %GLSLANG_DIR%
   cd  %GLSLANG_DIR%

   if not exist build32 (
       mkdir build32
   )
   if not exist build (
       mkdir build
   )

   echo Making 32-bit glslang
   echo *************************

   set GLSLANG_BUILD_DIR=%GLSLANG_DIR%\build32
   cd %GLSLANG_BUILD_DIR%

   echo Generating 32-bit Glslang CMake files for Visual Studio %VS_VERSION% -DCMAKE_INSTALL_PREFIX=install ..
   cmake -G "Visual Studio %VS_VERSION%" -DCMAKE_INSTALL_PREFIX=install ..

   echo Building 32-bit Glslang: MSBuild INSTALL.vcxproj /p:Platform=x86 /p:Configuration=Debug
   msbuild INSTALL.vcxproj /p:Platform=x86 /p:Configuration=Debug /verbosity:quiet

   REM Check for existence of one lib, even though we should check for all results
   if not exist %GLSLANG_BUILD_DIR%\glslang\Debug\glslangd.lib (
      echo.
      echo glslang 32-bit Debug build failed!
      set errorCode=1
   )
   echo Building Glslang: MSBuild INSTALL.vcxproj /p:Platform=x86 /p:Configuration=Release
   msbuild INSTALL.vcxproj /p:Platform=x86 /p:Configuration=Release /verbosity:quiet

   REM Check for existence of one lib, even though we should check for all results
   if not exist %GLSLANG_BUILD_DIR%\glslang\Release\glslang.lib (
      echo.
      echo glslang 32-bit Release build failed!
      set errorCode=1
   )

   cd ..

   echo Making 64-bit glslang
   echo *************************
   set GLSLANG_BUILD_DIR=%GLSLANG_DIR%\build
   cd %GLSLANG_BUILD_DIR%

   echo Generating 64-bit Glslang CMake files for Visual Studio %VS_VERSION% -DCMAKE_INSTALL_PREFIX=install ..
   cmake -G "Visual Studio %VS_VERSION% Win64" -DCMAKE_INSTALL_PREFIX=install ..

   echo Building 64-bit Glslang: MSBuild INSTALL.vcxproj /p:Platform=x64 /p:Configuration=Debug
   msbuild INSTALL.vcxproj /p:Platform=x64 /p:Configuration=Debug /verbosity:quiet

   REM Check for existence of one lib, even though we should check for all results
   if not exist %GLSLANG_BUILD_DIR%\glslang\Debug\glslangd.lib (
       echo.
       echo glslang 64-bit Debug build failed!
       set errorCode=1
   )

   echo Building Glslang: MSBuild INSTALL.vcxproj /p:Platform=x64 /p:Configuration=Release
   msbuild INSTALL.vcxproj /p:Platform=x64 /p:Configuration=Release /verbosity:quiet

   REM Check for existence of one lib, even though we should check for all results
   if not exist %GLSLANG_BUILD_DIR%\glslang\Release\glslang.lib (
       echo.
       echo glslang 64-bit Release build failed!
       set errorCode=1
   )
goto:eof

:create_jsoncpp
   echo.
   echo Creating local jsoncpp repository %JSONCPP_DIR%)
   mkdir %JSONCPP_DIR%
   cd %JSONCPP_DIR%
   git clone https://github.com/open-source-parsers/jsoncpp.git .
   git checkout %JSONCPP_REVISION%
   if not exist %JSONCPP_DIR%\include\json\json.h (
      echo jsoncpp source download failed!
      set errorCode=1
   )
goto:eof

:update_jsoncpp
   echo.
   echo Updating %JSONCPP_DIR%
   cd %JSONCPP_DIR%
   git fetch --all
   git checkout %JSONCPP_REVISION%
goto:eof

:build_jsoncpp
   echo.
   echo Building %JSONCPP_DIR%
   cd  %JSONCPP_DIR%
   python amalgamate.py

   if not exist %JSONCPP_DIR%\dist\json\json.h (
      echo.
      echo JsonCPP Amalgamation failed to generate %JSONCPP_DIR%\dist\json\json.h
      set errorCode=1
   )

REM    REM Cleanup any old directories lying around.
REM    if exist build32 (
REM       rmdir /s /q build32
REM    )
REM    if exist build (
REM       rmdir /s /q build
REM    )
REM
REM    echo Making 32-bit jsoncpp
REM    echo *************************
REM    mkdir build32
REM    set JSONCPP_BUILD_DIR=%JSONCPP_DIR%\build32
REM    cd %JSONCPP_BUILD_DIR%
REM
REM    echo Generating 32-bit JsonCPP CMake files for Visual Studio %VS_VERSION%
REM    cmake -G "Visual Studio %VS_VERSION%" .. -DMSVC_RUNTIME=static
REM
REM    echo Building 32-bit JsonCPP: MSBuild ALL_BUILD.vcxproj /p:Platform=x86 /p:Configuration=Debug
REM    msbuild ALL_BUILD.vcxproj /p:Platform=x86 /p:Configuration=Debug /verbosity:quiet
REM
REM    REM Check for existence of one lib, even though we should check for all results
REM    if not exist %JSONCPP_BUILD_DIR%\src\lib_json\Debug\jsoncpp.lib (
REM       echo.
REM       echo jsoncpp 32-bit Debug build failed!
REM       set errorCode=1
REM    )
REM    echo B Building 32-bit JsonCPP: MSBuild ALL_BUILD.vcxproj /p:Platform=x86 /p:Configuration=Release
REM    msbuild ALL_BUILD.vcxproj /p:Platform=x86 /p:Configuration=Release /verbosity:quiet
REM
REM    REM Check for existence of one lib, even though we should check for all results
REM    if not exist %JSONCPP_BUILD_DIR%\src\lib_json\Release\jsoncpp.lib (
REM       echo.
REM       echo jsoncpp 32-bit Release build failed!
REM       set errorCode=1
REM    )
REM
REM    cd ..
REM
REM    echo Making 64-bit jsoncpp
REM    echo *************************
REM    mkdir build
REM    set JSONCPP_BUILD_DIR=%JSONCPP_DIR%\build
REM    cd %JSONCPP_BUILD_DIR%
REM
REM    echo Generating 64-bit JsonCPP CMake files for Visual Studio %VS_VERSION%
REM    cmake -G "Visual Studio %VS_VERSION% Win64" .. -DMSVC_RUNTIME=static
REM
REM    echo Building 64-bit JsonCPP: MSBuild ALL_BUILD.vcxproj /p:Platform=x64 /p:Configuration=Debug
REM    msbuild ALL_BUILD.vcxproj /p:Platform=x64 /p:Configuration=Debug /verbosity:quiet
REM
REM    REM Check for existence of one lib, even though we should check for all results
REM    if not exist %JSONCPP_BUILD_DIR%\src\lib_json\Debug\jsoncpp.lib (
REM       echo.
REM       echo jsoncpp 64-bit Debug build failed!
REM       set errorCode=1
REM    )
REM    echo Building 64-bit JsonCPP: MSBuild ALL_BUILD.vcxproj /p:Platform=x64 /p:Configuration=Release
REM    msbuild ALL_BUILD.vcxproj /p:Platform=x64 /p:Configuration=Release /verbosity:quiet
REM
REM    REM Check for existence of one lib, even though we should check for all results
REM    if not exist %JSONCPP_BUILD_DIR%\src\lib_json\Release\jsoncpp.lib (
REM       echo.
REM       echo jsoncpp 64-bit Release build failed!
REM       set errorCode=1
REM    )
goto:eof
