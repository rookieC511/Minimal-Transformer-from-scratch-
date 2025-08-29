# check_env.py
import sys
import importlib
import pkg_resources
import subprocess

print("\n=== Python 解释器信息 ===")
print("Python 路径:", sys.executable)
print("Python 版本:", sys.version)

# 要检查的依赖
packages = ["transformers", "datasets", "accelerate"]

print("\n=== Hugging Face 相关依赖版本 ===")
for pkg in packages:
    try:
        module = importlib.import_module(pkg)
        print(f"{pkg} 版本: {module.__version__}")
        print(f"{pkg} 安装位置: {module.__file__}")
    except ImportError:
        print(f"{pkg} 未安装！")

print("\n=== pip 环境检测 ===")
# 列出已安装的版本，避免多版本冲突
try:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "list"], capture_output=True, text=True
    )
    for pkg in packages:
        for line in result.stdout.splitlines():
            if pkg.lower() in line.lower():
                print(line)
except Exception as e:
    print("无法执行 pip list:", e)

print("\n=== 检查完成 ===")
