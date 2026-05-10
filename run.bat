@echo off
chcp 65001 > nul

set ROOT=%~dp0
set IN=
set DBG=
set RES=

if "%~1"=="" goto run
if "%~1"=="--check" goto check
if "%~1"=="--help" goto help
if "%~1"=="-h" goto help
if "%~1"=="--auto" set IN=%ROOT%input_images
if "%~1"=="--debug" set DBG=--debug
if "%~1"=="--resume" set RES=--resume
if "%~1"=="--reset" set RES=--reset
if "%~1"=="--input" set IN=%~2
goto run

:run
where python >nul 2>nul
if errorlevel 1 (
    echo [FAIL] Python not found. Run install.ps1 as administrator.
    pause
    exit /b 1
)

if not defined IN set IN=%ROOT%input_images

if not exist "%ROOT%main.py" (
    echo [FAIL] main.py not found.
    pause
    exit /b 1
)

if not exist "%IN%" (
    echo [FAIL] Folder not found: %IN%
    pause
    exit /b 1
)

echo.
echo LibOCR starting...
echo Input: %IN%
echo.

python "%ROOT%main.py" --input "%IN%" %DBG% %RES%

if errorlevel 1 (
    echo.
    echo [FAIL] Processing error. Check logs\ocr.log
    pause
    exit /b 1
)

echo.
echo [OK] Done! Results in output_texts
pause
exit /b 0

:check
where python >nul 2>nul
if not errorlevel 1 (
    echo [OK] Python found:
    python --version
) else (
    echo [FAIL] Python not found
)
if exist "C:\Program Files\Tesseract-OCR\tesseract.exe" (
    echo [OK] Tesseract found
) else (
    echo [FAIL] Tesseract not found
)
echo.
echo Usage:
echo   run.bat              - process input_images
echo   run.bat --debug      - debug mode
echo   run.bat --check      - system check
echo   run.bat --help       - help
pause
exit /b 0

:help
echo LibOCR - OCR for library cards
echo.
echo Usage:
echo   run.bat              - process input_images
echo   run.bat --debug      - debug mode
echo   run.bat --check      - system check
echo   run.bat --help       - help
pause
exit /b 0