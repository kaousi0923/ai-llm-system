@echo off

:: 🧠 啟用 UTF-8 模式，避免 cp950 錯誤
set PYTHONUTF8=1
:: 🚫 關閉 TorchInductor，加快部署與避免 cl.exe 錯誤
set DISABLE_TORCH_INDUCTOR=1
set TORCHINDUCTOR_DISABLE=1
set TORCH_COMPILE=0

:: 定義虛擬環境名稱、Python 腳本和需求文件
set VENV_NAME=MIRDC_Unsloth_clone
set PYTHON_SCRIPT=main.py
set REQUIREMENTS=requirements3.txt

:: 檢查是否存在虛擬環境
conda info --envs | findstr /C:"%VENV_NAME% "
if errorlevel 1 (
    echo Creating virtual environment: %VENV_NAME%
    conda create -n %VENV_NAME% python=3.10 -y
    echo Virtual environment %VENV_NAME% already exists.
) else (
    echo Virtual environment %VENV_NAME% already exists.
)

:: 啟用虛擬環境
CALL conda activate %VENV_NAME%

:: 檢查並安裝 requirements
python -m pip install --disable-pip-version-check --no-cache-dir --exists-action i -r %REQUIREMENTS%

:: 執行 Python 腳本
python %PYTHON_SCRIPT%

:: 終止虛擬環境
CALL conda deactivate

echo Script execution completed.
pause