@echo off
chcp 65001 > nul

set "ROOT=%~dp0"
set "IN="
set "DBG="
set "RES="

call :parse_args %*
goto :run

:parse_args
if "%~1"=="" exit /b 0
if "%~1"=="--auto" set "IN=%ROOT%input_images" & exit /b 0
if "%~1"=="--input" set "IN=%~2" & exit /b 0
if "%~1"=="--debug" set "DBG=--debug" & exit /b 0
if "%~1"=="--resume" set "RES=--resume" & exit /b 0
if "%~1"=="--reset" set "RES=--reset" & exit /b 0
if "%~1"=="--check" goto :check
if "%~1"=="--help" goto :help
if "%~1"=="-h" goto :help
shift
goto :parse_args

:run
where python >nul 2>nul
if errorlevel 1 (
    echo [FAIL] Python не найден.
    echo Запустите install.ps1 от имени администратора.
    echo.
    pause
    exit /b 1
)

if not defined IN set "IN=%ROOT%input_images"

set "MAIN=%ROOT%main.py"
if not exist "%MAIN%" (
    echo [FAIL] main.py не найден.
    echo.
    pause
    exit /b 1
)

if not exist "%IN%" (
    echo [FAIL] Папка не найдена: %IN%
    echo.
    pause
    exit /b 1
)

echo.
echo [....] Запуск LibOCR...
echo     Вход: %IN%
echo.

python "%MAIN%" --input "%IN%" %DBG% %RES%

if errorlevel 1 (
    echo.
    echo [FAIL] Ошибка обработки. Проверьте logs\ocr.log
    echo.
    pause
    exit /b 1
)

echo.
echo [ OK ] Готово! Результаты в output_texts
echo.
pause
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
echo   run.bat               - обработать input_images
echo   run.bat --debug       - режим отладки
echo   run.bat --check       - проверка системы
echo   run.bat --help        - эта справка
echo.
pause
exit /b 0

:help
echo LibOCR - OCR для библиотечных карточек
echo.
echo Использование:
echo   run.bat               - обработать input_images
echo   run.bat --debug       - режим отладки
echo   run.bat --check       - проверка системы
echo   run.bat --help        - эта справка
echo.
pause
exit /b 0