@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion

set "ROOT=%~dp0"
set "IN="
set "DBG="
set "RES="

:parse_args
if "%~1"=="" goto :run
if "%~1"=="--auto" set "IN=%ROOT%input_images" & shift & goto :parse_args
if "%~1"=="--input" set "IN=%~2" & shift & shift & goto :parse_args
if "%~1"=="--debug" set "DBG=--debug" & shift & goto :parse_args
if "%~1"=="--resume" set "RES=--resume" & shift & goto :parse_args
if "%~1"=="--reset" set "RES=--reset" & shift & goto :parse_args
if "%~1"=="--check" goto :check
if "%~1"=="--help" goto :help
if "%~1"=="-h" goto :help
shift & goto :parse_args

:run
where python >nul 2>nul
if errorlevel 1 (
    echo [FAIL] Python не найден.
    echo Запустите install.ps1 от имени администратора
    echo.
    echo Нажмите любую клавишу...
    pause > nul
    exit /b 1
)

if not defined IN set "IN=%ROOT%input_images"

set "MAIN=%ROOT%main.py"
if not exist "%MAIN%" (
    echo [FAIL] main.py не найден
    echo.
    echo Нажмите любую клавишу...
    pause > nul
    exit /b 1
)
if not exist "%IN%" (
    echo [FAIL] Папка не найдена: %IN%
    echo.
    echo Нажмите любую клавишу...
    pause > nul
    exit /b 1
)

echo.
echo [....] Запуск LibOCR...
echo     Вход:  %IN%
echo.

python "%MAIN%" --input "%IN%" %DBG% %RES%
set "PY_EXIT=%errorlevel%"

if %PY_EXIT% neq 0 (
    echo.
    echo [FAIL] Ошибка обработки (%PY_EXIT%). Проверьте logs\ocr.log
    echo.
    echo Нажмите любую клавишу...
    pause > nul
    exit /b %PY_EXIT%
)

echo.
echo [ OK ] Готово! Результаты в output_texts\
echo.
echo Нажмите любую клавишу...
pause > nul
exit /b 0

:check
where python >nul 2>nul
if not errorlevel 1 (
    echo [ OK ] Python найден:
    python --version
) else (
    echo [FAIL] Python не найден
)
if exist "C:\Program Files\Tesseract-OCR\tesseract.exe" (
    echo [ OK ] Tesseract найден
) else (
    echo [FAIL] Tesseract не найден
)
echo.
echo Использование:
echo   run.bat               — обработать input_images
echo   run.bat --auto        — обработать input_images
echo   run.bat --debug       — режим отладки
echo   run.bat --input path   — указать папку
echo   run.bat --resume       — продолжить batch
echo   run.bat --reset       — сбросить batch
echo   run.bat --check       — проверка системы
echo   run.bat --help        — эта справка
echo.
echo Нажмите любую клавишу...
pause > nul
exit /b 0

:help
echo LibOCR — OCR для библиотечных карточек
echo.
echo Использование:
echo   run.bat               — обработать input_images
echo   run.bat --auto        — обработать input_images
echo   run.bat --debug       — режим отладки
echo   run.bat --input path   — указать папку
echo   run.bat --resume       — продолжить batch
echo   run.bat --reset       — сбросить batch
echo   run.bat --check       — проверка системы
echo   run.bat --help        — эта справка
echo.
echo Нажмите любую клавишу...
pause > nul
exit /b 0
