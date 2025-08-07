@echo off
echo ========================================
echo   AI/LLM RAG System - Test Runner
echo ========================================
echo.

:: 設定 UTF-8 編碼
chcp 65001 > nul
set PYTHONUTF8=1

:: 設定虛擬環境名稱
set VENV_NAME=MIRDC_Unsloth_clone

:: 檢查虛擬環境是否存在
conda info --envs | findstr /C:"%VENV_NAME% " > nul
if errorlevel 1 (
    echo [錯誤] 虛擬環境 %VENV_NAME% 不存在！
    echo 請先執行 llm_install.bat 安裝環境
    pause
    exit /b 1
)

:: 啟用虛擬環境
echo [步驟 1/4] 啟用虛擬環境...
call conda activate %VENV_NAME%

:: 安裝測試依賴
echo [步驟 2/4] 檢查測試套件...
pip show pytest > nul 2>&1
if errorlevel 1 (
    echo 安裝測試套件...
    pip install -r requirements-test.txt
)

:: 選擇測試模式
echo.
echo 請選擇測試模式：
echo 1. 執行所有測試
echo 2. 執行單元測試 (快速)
echo 3. 執行 API 測試
echo 4. 執行整合測試
echo 5. 生成測試覆蓋率報告
echo 6. 執行特定測試檔案
echo 0. 退出
echo.
set /p choice="請輸入選項 (0-6): "

if "%choice%"=="0" goto end
if "%choice%"=="1" goto all_tests
if "%choice%"=="2" goto unit_tests
if "%choice%"=="3" goto api_tests
if "%choice%"=="4" goto integration_tests
if "%choice%"=="5" goto coverage
if "%choice%"=="6" goto specific_test

:all_tests
echo.
echo [步驟 3/4] 執行所有測試...
pytest -v --tb=short
goto end

:unit_tests
echo.
echo [步驟 3/4] 執行單元測試...
pytest -v -m unit --tb=short
goto end

:api_tests
echo.
echo [步驟 3/4] 執行 API 測試...
pytest -v -m api --tb=short
goto end

:integration_tests
echo.
echo [步驟 3/4] 執行整合測試...
pytest -v -m integration --tb=short
goto end

:coverage
echo.
echo [步驟 3/4] 生成測試覆蓋率報告...
pytest --cov=. --cov-report=html --cov-report=term
echo.
echo 覆蓋率報告已生成在 htmlcov/index.html
echo 正在開啟報告...
start htmlcov/index.html
goto end

:specific_test
echo.
set /p testfile="請輸入測試檔案名稱 (例如: test_api.py): "
echo [步驟 3/4] 執行測試檔案 %testfile%...
pytest tests/%testfile% -v --tb=short
goto end

:end
echo.
echo [步驟 4/4] 清理環境...
call conda deactivate

echo.
echo ========================================
echo   測試執行完成！
echo ========================================
pause