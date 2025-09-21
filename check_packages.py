import sys
import site

print("🐍 Python executable being used:")
print(sys.executable)  # Shows venv Python path

print("\n📂 Site-packages path:")
print(site.getsitepackages())  # Shows where pip installs packages

print("\n✅ Checking installed packages in this microservice:")

# Check FastAPI
try:
    import fastapi
    print(f"✅ fastapi → {fastapi.__file__}")
    print(f"   Version: {fastapi.__version__}")
except ImportError:
    print("❌ fastapi not installed")

# Check Uvicorn
try:
    import uvicorn
    print(f"✅ uvicorn → {uvicorn.__file__}")
    print(f"   Version: {uvicorn.__version__}")
except ImportError:
    print("❌ uvicorn not installed")

# Check PyPDF2
try:
    import PyPDF2
    print(f"✅ PyPDF2 → {PyPDF2.__file__}")
    print(f"   Version: {PyPDF2.__version__}")
except ImportError:
    print("❌ PyPDF2 not installed")

# Check Pydantic
try:
    import pydantic
    print(f"✅ pydantic → {pydantic.__file__}")
    print(f"   Version: {pydantic.__version__}")
except ImportError:
    print("❌ pydantic not installed")

print("\n🔧 Virtual Environment Status:")
if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
    print("✅ Running in virtual environment")
    print(f"   Virtual env path: {sys.prefix}")
    print(f"   Base Python path: {sys.base_prefix}")
else:
    print("⚠️  Not running in virtual environment")

print("\n📋 All installed packages:")
import pkg_resources
installed_packages = [d for d in pkg_resources.working_set]
for package in sorted(installed_packages, key=lambda x: x.project_name.lower()):
    print(f"   {package.project_name} → {package.version}")
