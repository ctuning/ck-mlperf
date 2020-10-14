@echo off

rem
rem CK installation script
rem
rem See CK LICENSE for licensing details.
rem See CK COPYRIGHT for copyright details.
rem

rem PACKAGE_DIR
rem INSTALL_DIR
rem PYTHON_PACKAGE_NAME
rem PIP_INSTALL_OPTIONS

rem This is where pip will install the modules.
rem It has its own funny structure we don't control :
rem 

set EXTRA_PYTHON_SITE=%INSTALL_DIR%\build

echo **************************************************************
echo.
echo Cleanup: removing %EXTRA_PYTHON_SITE%
if exist "%EXTRA_PYTHON_SITE%" (
 rmdir /S /Q "%EXTRA_PYTHON_SITE%"
)

rem ######################################################################################
echo.
echo Installing %PYTHON_PACKAGE_NAME% and its dependencies to %PACKAGE_LIB_DIR% ...

%CK_ENV_COMPILER_PYTHON_FILE% -m pip install --ignore-installed %PYTHON_PACKAGE_NAME% -t %EXTRA_PYTHON_SITE% %PIP_INSTALL_OPTIONS%

if %errorlevel% neq 0 (
 echo.
 echo Error: installation failed!
 goto err
)


set POST_INSTALL_SCRIPT=%ORIGINAL_PACKAGE_DIR%\post-install.bat
if EXIST "%POST_INSTALL_SCRIPT%" (
  echo.
  echo Executing post install script %POST_INSTALL_SCRIPT% ...
  echo.

  call %POST_INSTALL_SCRIPT%

  if %errorlevel% neq 0 (
   echo.
   echo Error: Failed executing post install script %POST_INSTALL_SCRIPT% ...
   goto err
  )
)

exit /b 0

:err
exit /b 1

