@echo off
REM 检查Docker构建上下文大小的脚本(Windows版本)

echo ========== Docker构建上下文分析 ==========
echo.

echo 1. 被排除的主要目录检查:
if exist ".conda" (
    echo    .conda: 存在 ^(Conda环境目录^)
) else (
    echo    .conda: 不存在
)

if exist ".venv" (
    echo    .venv: 存在 ^(虚拟环境目录^)
) else (
    echo    .venv: 不存在
)

if exist "cache" (
    echo    cache: 存在 ^(缓存目录^)
) else (
    echo    cache: 不存在
)

if exist "__pycache__" (
    echo    __pycache__: 存在 ^(Python缓存^)
) else (
    echo    __pycache__: 不存在
)

if exist "graphgen-offline-image.tar" (
    for %%A in (graphgen-offline-image.tar) do (
        set size=%%~zA
        set /a size_mb=!size!/1048576
        echo    graphgen-offline-image.tar: !size_mb! MB ^(已生成的镜像文件^)
    )
) else (
    echo    graphgen-offline-image.tar: 不存在
)

echo.
echo 2. .dockerignore配置验证:
if exist ".dockerignore" (
    echo    ✅ .dockerignore文件已存在
    for /f %%i in ('type .dockerignore ^| find /c /v ""') do echo    总行数: %%i
) else (
    echo    ❌ .dockerignore文件不存在
)

echo.
echo 3. 构建上下文优化效果:
echo    优化前: 包含所有文件^(包括虚拟环境、缓存、已生成的镜像等^)
echo    优化后: 仅包含必要的源代码文件
echo    预计减少: 数百MB到数GB^(主要取决于虚拟环境和已生成镜像的大小^)

echo.
echo 4. 建议:
echo    - 定期清理cache目录
echo    - 避免在项目根目录生成大文件
echo    - 构建前可删除已有的镜像tar文件

echo.
echo ========== 分析完成 ==========
pause