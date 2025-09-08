# GraphGen PowerShell启动脚本 - 从.env文件加载
param(
    [string]$OutputDir = "cache"
)

# 检查.env文件是否存在
if (-not (Test-Path ".env")) {
    Write-Host "错误：找不到.env文件，请确保在GraphGen项目根目录中运行此脚本" -ForegroundColor Red
    exit 1
}

# 从.env文件加载环境变量
Write-Host "正在从.env文件加载环境变量..." -ForegroundColor Yellow
Get-Content ".env" | ForEach-Object {
    if ($_ -match "^\s*([^#][^=]*)\s*=\s*(.*)\s*$") {
        $name = $matches[1].Trim()
        $value = $matches[2].Trim()
        # 移除可能的引号
        $value = $value -replace '^["'']|["'']$', ''
        Set-Item -Path "env:$name" -Value $value
        Write-Host "设置 $name = $value" -ForegroundColor Green
    }
}

Write-Host "环境变量设置完成" -ForegroundColor Green

# 检查必要的环境变量
$requiredVars = @("SYNTHESIZER_MODEL", "SYNTHESIZER_BASE_URL", "SYNTHESIZER_API_KEY", "TRAINEE_MODEL", "TRAINEE_BASE_URL", "TRAINEE_API_KEY")
$missingVars = @()

foreach ($var in $requiredVars) {
    if (-not (Get-Item -Path "env:$var" -ErrorAction SilentlyContinue)) {
        $missingVars += $var
    }
}

if ($missingVars.Count -gt 0) {
    Write-Host "错误：缺少必要的环境变量：$($missingVars -join ', ')" -ForegroundColor Red
    exit 1
}

# 检查graphg是否已安装
try {
    $null = Get-Command graphg -ErrorAction Stop
    Write-Host "GraphGen已安装" -ForegroundColor Green
} catch {
    Write-Host "正在安装GraphGen..." -ForegroundColor Yellow
    uv pip install graphg
    if ($LASTEXITCODE -ne 0) {
        Write-Host "GraphGen安装失败，请检查网络连接和uv配置" -ForegroundColor Red
        exit 1
    }
    Write-Host "GraphGen安装成功" -ForegroundColor Green
}

# 运行GraphGen
Write-Host "正在启动GraphGen，输出目录：$OutputDir" -ForegroundColor Green
graphg --output_dir $OutputDir