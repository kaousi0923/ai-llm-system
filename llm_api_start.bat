@echo off
:: 切換到 .bat 檔案所在的目錄，避免相對路徑亂指
cd /d "%~dp0"

:: 🧠 啟用 UTF-8 模式，避免 cp950 錯誤
set PYTHONUTF8=1

:: 🚫 關閉 TorchInductor，加快部署與避免 cl.exe 錯誤
set DISABLE_TORCH_INDUCTOR=1
set TORCHINDUCTOR_DISABLE=1
set TORCH_COMPILE=0

:: 💡 禁用 hf_transfer 避免卡在 99%
set HF_HUB_ENABLE_HF_TRANSFER=0

:: 📁 設定 Hugging Face 模型快取目錄到 bat 同層的 base_models 資料夾
set SCRIPT_DIR=%~dp0
set HUGGINGFACE_HUB_CACHE=%SCRIPT_DIR%base_models

:: 🚀 定義虛擬環境名稱、Python 腳本與 requirements
set VENV_NAME=MIRDC_Unsloth_clone
set PYTHON_SCRIPT=main.py
set REQUIREMENTS=requirements2.txt

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

:: 執行 Python 腳本
python %PYTHON_SCRIPT%

:: 終止虛擬環境
CALL conda deactivate

echo Script execution completed.
pause