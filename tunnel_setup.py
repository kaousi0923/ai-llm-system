# -*- coding: utf-8 -*-
"""
Created on Tue May 14 15:05:44 2024

@author: m112719
"""

import os
import subprocess
import urllib.request
import zipfile

# 定義下載URL和本地文件名
cloudflared_url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
bore_url = "https://github.com/ekzhang/bore/releases/download/v0.5.0/bore-v0.5.0-x86_64-pc-windows-msvc.zip"

cloudflared_path = os.path.join(os.getcwd(),'dependencies', "cloudflared.exe")
bore_zip_path = os.path.join(os.getcwd(),'dependencies', "bore.zip")
bore_executable_path = os.path.join(os.getcwd(),'dependencies',"bore.exe")

# 下載文件
def download_file(url, path):
    print(f"Downloading {url} to {path}")
    urllib.request.urlretrieve(url, path)
    print("Download complete")

# 解壓bore
def unzip_file(zip_path, extract_to):
    print(f"Unzipping {zip_path}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Unzip complete")

# 下載cloudflared
if not os.path.exists(cloudflared_path):
    download_file(cloudflared_url, cloudflared_path)

# 下載並解壓bore
if not os.path.exists(bore_executable_path):
    download_file(bore_url, bore_zip_path)
    unzip_file(bore_zip_path, os.getcwd())

# 取得連接埠號碼，預設為5000
port = os.getenv('TUNNEL_PORT', '5000')

# 啟動cloudflared隧道
def start_cloudflared():
    print(f"Starting cloudflared tunnel on port {port}...")
    subprocess.Popen([cloudflared_path, "tunnel", "--url", f"http://localhost:{port}"])

# 啟動bore隧道
def start_bore():
    print("Starting bore tunnel on port {port}...")
    subprocess.Popen([bore_executable_path, "local", port, "--to", "bore.pub"])
    
def main():
    start_cloudflared()
    start_bore()
    
if __name__ == "__main__":
    main()