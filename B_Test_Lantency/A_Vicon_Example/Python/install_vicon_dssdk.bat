::Install api into Python site packages directory (e.g. C:\Python27\Lib\site-packages)
pip install "%~dp0vicon_dssdk"

:: Prompt for press any key to continue if we're not running from a command prompt
@echo %cmdcmdline% | findstr /l "\"\"" >NUL 2>&1
@if %errorlevel% EQU 0 pause