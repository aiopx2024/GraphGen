# GraphGen 批量处理 PowerShell 脚本
# 用于Windows环境下批量处理txt文件并生成各种类型的语料对

param(
    [Parameter(Mandatory=$true)]
    [string]$InputDir,           # txt文件目录
    
    [Parameter(Mandatory=$true)]
    [string]$OutputDir,          # 输出目录
    
    [string[]]$Types = @("atomic", "aggregated", "multi_hop"),  # 生成的语料类型
    
    [int]$ChunkSize = 512,       # 文本分块大小
    
    [int]$BatchSize = 10,        # 每批次处理的文件数
    
    [switch]$NoTrainee           # 禁用trainee模型
)

Write-Host "🚀 GraphGen 批量处理工具" -ForegroundColor Green
Write-Host "=" * 50

# 检查输入目录
if (-not (Test-Path $InputDir)) {
    Write-Host "❌ 输入目录不存在: $InputDir" -ForegroundColor Red
    exit 1
}

# 创建输出目录
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

# 设置环境变量（从.env文件读取）
if (Test-Path ".env") {
    Write-Host "📝 从.env文件加载环境变量..." -ForegroundColor Yellow
    Get-Content ".env" | ForEach-Object {
        if ($_ -match "^([^#=]+)=(.*)$") {
            $name = $matches[1].Trim()
            $value = $matches[2].Trim().Trim('"')
            Set-Item -Path "env:$name" -Value $value
            Write-Host "设置 $name = $value" -ForegroundColor Green
        }
    }
} else {
    Write-Host "⚠️  未找到.env文件，请确保已设置必要的环境变量" -ForegroundColor Yellow
}

# 检查必要的环境变量
$requiredVars = @("SYNTHESIZER_MODEL", "SYNTHESIZER_BASE_URL", "SYNTHESIZER_API_KEY")
if (-not $NoTrainee) {
    $requiredVars += @("TRAINEE_MODEL", "TRAINEE_BASE_URL", "TRAINEE_API_KEY")
}

$missingVars = @()
foreach ($var in $requiredVars) {
    if (-not (Get-Item -Path "env:$var" -ErrorAction SilentlyContinue)) {
        $missingVars += $var
    }
}

if ($missingVars.Count -gt 0) {
    Write-Host "❌ 缺少必要的环境变量：$($missingVars -join ', ')" -ForegroundColor Red
    Write-Host "请在.env文件中设置这些变量或手动设置环境变量" -ForegroundColor Yellow
    exit 1
}

# 查找txt文件
$txtFiles = Get-ChildItem -Path $InputDir -Filter "*.txt" -File
if ($txtFiles.Count -eq 0) {
    Write-Host "❌ 在目录 $InputDir 中未找到txt文件" -ForegroundColor Red
    exit 1
}

Write-Host "📁 找到 $($txtFiles.Count) 个txt文件" -ForegroundColor Green
Write-Host "📋 将生成的语料类型: $($Types -join ', ')" -ForegroundColor Green

# 构建Python命令参数
$pythonArgs = @(
    "batch_process.py",
    "--input-dir", "`"$InputDir`"",
    "--output-dir", "`"$OutputDir`"",
    "--types", ($Types -join " "),
    "--chunk-size", $ChunkSize,
    "--batch-size", $BatchSize
)

if ($NoTrainee) {
    $pythonArgs += "--no-trainee"
}

# 显示将要执行的命令
Write-Host "🔄 执行命令:" -ForegroundColor Yellow
Write-Host "python $($pythonArgs -join ' ')" -ForegroundColor Cyan

# 执行批处理
try {
    & python @pythonArgs
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "🎉 批量处理完成！" -ForegroundColor Green
        Write-Host "📂 结果保存在: $OutputDir" -ForegroundColor Green
        
        # 显示输出目录结构
        Write-Host "`n📊 输出目录结构:" -ForegroundColor Yellow
        Get-ChildItem -Path $OutputDir -Recurse | ForEach-Object {
            $relativePath = $_.FullName.Substring($OutputDir.Length)
            if ($_.PSIsContainer) {
                Write-Host "📁 $relativePath" -ForegroundColor Cyan
            } else {
                Write-Host "📄 $relativePath" -ForegroundColor White
            }
        }
    } else {
        Write-Host "❌ 批量处理失败" -ForegroundColor Red
        exit $LASTEXITCODE
    }
} catch {
    Write-Host "❌ 执行过程中出现错误: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

Write-Host "`n✅ 所有任务完成！" -ForegroundColor Green