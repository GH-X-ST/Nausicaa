::Uninstall api from site packages
pip uninstall -y vicon_dssdk

:: Prompt for press any key to continue if we're not running from a command prompt
@echo %cmdcmdline% | findstr /l "\"\"" >NUL 2>&1
@if %errorlevel% EQU 0 pause
