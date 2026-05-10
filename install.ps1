#Requires -RunAsAdministrator

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonVersion = "3.11.9"
$PythonInstaller = "python-3.11.9-amd64.exe"
$PythonUrl = "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe"
$PythonInstallDir = Join-Path $ScriptDir "libraries"
$TesseractUrl = "https://github.com/UB-Mannheim/tesseract/releases/download/v5.3.3.20240505/tesseract-ocr-w64-setup-5.3.3.20240505.exe"
$TesseractInstallDir = "C:\Program Files\Tesseract-OCR"
$TessdataUrl = "https://github.com/tesseract-ocr/tessdata_best/raw/main/rus.traineddata"
$MaxRetries = 3

function Write-ColorOutput($Message, $Color) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $Color
    Write-Output $Message
    $host.UI.RawUI.ForegroundColor = $fc
}

function Download-WithRetry {
    param([string]$Url, [string]$OutputPath, [string]$Description)

    Write-ColorOutput "[....] Скачивание: $Description" "Cyan"

    for ($i = 1; $i -le $MaxRetries; $i++) {
        try {
            Invoke-WebRequest -Uri $Url -OutFile $OutputPath -ErrorAction Stop
            Write-ColorOutput "[ OK ] Скачано" "Green"
            return $true
        }
        catch {
            if ($i -lt $MaxRetries) {
                $wait = [Math]::Pow(2, $i)
                Write-ColorOutput "[WARN] Попытка $i не удалась. Повтор через $wait сек..." "Yellow"
                Start-Sleep -Seconds $wait
            }
            else {
                Write-ColorOutput "[FAIL] Не удалось скачать: $Description" "Red"
                return $false
            }
        }
    }
}

Write-ColorOutput "============================================" "White"
Write-ColorOutput "  Установка LibOCR" "White"
Write-ColorOutput "============================================" "White"
Write-ColorOutput ""

$dirs = @("input_images", "output_texts", "logs", "debug", "temp_install")
foreach ($dir in $dirs) {
    $path = Join-Path $ScriptDir $dir
    if (-not (Test-Path $path)) {
        New-Item -ItemType Directory -Force -Path $path | Out-Null
    }
}

$pythonExe = Join-Path $PythonInstallDir "python.exe"
$pipExe = Join-Path $PythonInstallDir "Scripts\pip.exe"

$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (Test-Path $pythonExe) {
    Write-ColorOutput "[ OK ] Python уже установлен в libraries" "Green"
}
else {
    Write-ColorOutput "[....] Установка Python $PythonVersion..." "Cyan"

    $tempDir = Join-Path $ScriptDir "temp_install"
    if (-not (Test-Path $tempDir)) {
        New-Item -ItemType Directory -Force -Path $tempDir | Out-Null
    }

    $installerPath = Join-Path $tempDir $PythonInstaller

    if (Download-WithRetry -Url $PythonUrl -OutputPath $installerPath -Description "Python $PythonVersion") {
        if (-not $isAdmin) {
            Write-ColorOutput "[WARN] Установка Python требует прав администратора" "Yellow"
            Write-ColorOutput "  Установите Python вручную: https://www.python.org/downloads/" "Yellow"
            Write-ColorOutput "  Или запустите PowerShell от имени администратора" "Yellow"
        }
        else {
            Write-ColorOutput "[....] Установка Python..." "Cyan"

            $proc = Start-Process -FilePath $installerPath `
                -ArgumentList "/quiet", "InstallAllUsers=0", "TargetDir=$PythonInstallDir", `
                            "Include_pip=1", "Include_test=0", "PrependPath=1" `
                -Wait -PassThru -NoNewWindow

            if ($proc.ExitCode -ne 0) {
                Write-ColorOutput "[WARN] Автоматическая установка не удалась" "Yellow"
            }
            else {
                Write-ColorOutput "[ OK ] Python установлен" "Green"
            }
        }
    }
}

if (Test-Path $pythonExe) {
    Write-ColorOutput "[....] Обновление pip..." "Cyan"
    & $pythonExe -m pip install --upgrade pip --quiet 2>&1 | Out-Null

    $requirementsPath = Join-Path $ScriptDir "requirements.txt"

    if (Test-Path $requirementsPath) {
        Write-ColorOutput "[....] Установка библиотек..." "Cyan"
        & $pipExe install -r $requirementsPath 2>&1 | Out-Null

        if ($LASTEXITCODE -ne 0) {
            Write-ColorOutput "[WARN] Некоторые библиотеки не установились" "Yellow"
        }
        else {
            Write-ColorOutput "[ OK ] Библиотеки установлены" "Green"
        }
    }
}
else {
    Write-ColorOutput "[FAIL] Python не найден. Установите вручную." "Red"
}

$tesseractExe = Join-Path $TesseractInstallDir "tesseract.exe"

if (Test-Path $tesseractExe) {
    Write-ColorOutput "[ OK ] Tesseract уже установлен" "Green"
}
else {
    if (-not $isAdmin) {
        Write-ColorOutput "[WARN] Установка Tesseract требует прав администратора" "Yellow"
        Write-ColorOutput "  Скачайте вручную: https://github.com/UB-Mannheim/tesseract/releases" "Yellow"
        Write-ColorOutput "  Или запустите PowerShell от имени администратора" "Yellow"
    }
    else {
        Write-ColorOutput "[....] Установка Tesseract..." "Cyan"

        $tempDir = Join-Path $ScriptDir "temp_install"
        $tesseractInstallerPath = Join-Path $tempDir "tesseract-installer.exe"

        if (Download-WithRetry -Url $TesseractUrl -OutFile $tesseractInstallerPath -Description "Tesseract") {
            Write-ColorOutput "[....] Запуск установщика Tesseract..." "Cyan"

            $proc = Start-Process -FilePath $tesseractInstallerPath `
                -ArgumentList "/SILENT", "/NORESTART" `
                -Wait -PassThru -NoNewWindow

            Start-Sleep -Seconds 5

            if (Test-Path $tesseractExe) {
                Write-ColorOutput "[ OK ] Tesseract установлен" "Green"
            }
            else {
                Write-ColorOutput "[WARN] Tesseract не установился автоматически" "Yellow"
                Write-ColorOutput "  Скачайте вручную: https://github.com/UB-Mannheim/tesseract/releases" "Yellow"
            }
        }
    }
}

$tessdataPath = Join-Path $TesseractInstallDir "tessdata\rus.traineddata"

if (Test-Path $tessdataPath) {
    Write-ColorOutput "[ OK ] rus.traineddata уже есть" "Green"
}
else {
    Write-ColorOutput "[....] Скачивание rus.traineddata..." "Cyan"

    $tessdataDir = Join-Path $TesseractInstallDir "tessdata"
    if (-not (Test-Path $tessdataDir)) {
        if ($isAdmin) {
            New-Item -ItemType Directory -Force -Path $tessdataDir | Out-Null
        }
        else {
            Write-ColorOutput "[WARN] Нет прав для создания tessdata. Скачайте rus.traineddata вручную." "Yellow"
        }
    }

    if (Test-Path $tessdataDir) {
        if (Download-WithRetry -Url $TessdataUrl -OutputPath $tessdataPath -Description "rus.traineddata") {
            Write-ColorOutput "[ OK ] rus.traineddata скачан" "Green"
        }
    }
    else {
        Write-ColorOutput "[WARN] rus.traineddata не скачан. Скачайте вручную в папку tessdata." "Yellow"
    }
}

$testImages = Get-ChildItem -Path (Join-Path $ScriptDir "input_images") -Filter *.jpg,*.png -ErrorAction SilentlyContinue | Select-Object -First 1

if ($testImages -and (Test-Path $tesseractExe)) {
    Write-ColorOutput "[....] Тестирование OCR..." "Cyan"

    $testOutput = Join-Path $ScriptDir "temp_install\test_output"

    & $tesseractExe $testImages.FullName $testOutput -l rus+eng --oem 3 --psm 6 2>&1 | Out-Null

    if (Test-Path "$testOutput.txt") {
        $testText = Get-Content "$testOutput.txt" -Raw -Encoding UTF8
        Write-ColorOutput "[ OK ] Тест успешен (распознано символов: $($testText.Length))" "Green"
    }
    else {
        Write-ColorOutput "[WARN] Тест не дал результатов" "Yellow"
    }
}

Write-ColorOutput "[....] Очистка..." "Cyan"
Remove-Item -Path (Join-Path $ScriptDir "temp_install") -Recurse -Force -ErrorAction SilentlyContinue

Write-ColorOutput "" "White"
Write-ColorOutput "============================================" "Green"
Write-ColorOutput "  Установка завершена!" "Green"
Write-ColorOutput "============================================" "Green"
Write-ColorOutput ""
Write-ColorOutput "Теперь запустите run.bat" "White"
Write-ColorOutput ""

pause
