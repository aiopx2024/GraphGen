# GraphGen PowerShell启动脚本
# 设置环境变量
$env:SYNTHESIZER_MODEL = "Qwen/Qwen2.5-7B-Instruct"
$env:SYNTHESIZER_BASE_URL = "https://api.siliconflow.cn/v1"
$env:SYNTHESIZER_API_KEY = "sk-mmjqwndvcovgaxrezdtwnzvhpqajkjqrnsifaqrxifxoxvce"
$env:TRAINEE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
$env:TRAINEE_BASE_URL = "https://api.siliconflow.cn/v1"
$env:TRAINEE_API_KEY = "sk-mmjqwndvcovgaxrezdtwnzvhpqajkjqrnsifaqrxifxoxvce"

Write-Host "环境变量已设置完成" -ForegroundColor Green
Write-Host "SYNTHESIZER_MODEL: $env:SYNTHESIZER_MODEL" -ForegroundColor Yellow
Write-Host "TRAINEE_MODEL: $env:TRAINEE_MODEL" -ForegroundColor Yellow

# 检查graphg是否已安装
try {
    $version = graphg --version 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "GraphGen已安装，版本: $version" -ForegroundColor Green
    } else {
        throw "GraphGen未安装"
    }
} catch {
    Write-Host "正在安装GraphGen..." -ForegroundColor Yellow
    uv pip install graphg
    if ($LASTEXITCODE -eq 0) {
        Write-Host "GraphGen安装成功" -ForegroundColor Green
    } else {
        Write-Host "GraphGen安装失败，请检查网络连接和uv配置" -ForegroundColor Red
        exit 1
    }
}

# 运行GraphGen
Write-Host "正在启动GraphGen..." -ForegroundColor Green
graphg --output_dir cache