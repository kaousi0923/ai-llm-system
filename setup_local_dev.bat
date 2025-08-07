@echo off
:: RAG系統 - 本機開發環境快速設置腳本
cd /d "%~dp0"

echo ========================================
echo 🛠️  RAG 系統本機開發環境設置
echo ========================================
echo.

:: 檢查 .env.example 是否存在
if not exist ".env.example" (
    echo ❌ 錯誤: .env.example 文件不存在
    pause
    exit /b 1
)

:: 檢查 .env 是否已存在
if exist ".env" (
    echo 📋 發現現有的 .env 文件
    echo.
    set /p choice="是否要重新創建 .env 文件? (y/N): "
    if /i not "%choice%"=="y" (
        echo 💡 保持現有配置，你可以手動編輯 .env 文件
        goto :show_next_steps
    )
)

:: 複製 .env.example 到 .env
copy ".env.example" ".env" >nul
echo ✅ 已創建 .env 文件

echo.
echo 📝 接下來你需要編輯 .env 文件並填入正確的 API 密鑰:
echo.
echo   OPENAI_API_KEY=你的OpenAI密鑰
echo   HF_TOKEN=你的HuggingFace密鑰  
echo   AUTHORIZATION_TOKEN=你的授權密鑰
echo.

set /p edit_now="是否現在就打開 .env 文件進行編輯? (Y/n): "
if /i not "%edit_now%"=="n" (
    notepad ".env"
)

:show_next_steps
echo.
echo ========================================
echo 🎯 下一步操作指南:
echo ========================================
echo.
echo 1. 📝 確保 .env 文件中的 API 密鑰正確
echo 2. 🚀 執行 llm_api_start.bat 啟動系統
echo 3. 🌐 訪問 http://localhost:5000/docs 查看 API 文檔
echo 4. 🐳 可選: 使用 Docker 部署 (docker/docker-deploy.bat)
echo.
echo ========================================
echo 💡 提示: .env 文件已被 Git 忽略，不會推送到 GitHub
echo ========================================
echo.
pause