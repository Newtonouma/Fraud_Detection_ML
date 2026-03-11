"""
build_exe.py  —  v2.0
======================
Packages app.py + fraud_pipeline.py + merged_fraud_dataset.csv into one .exe

SETUP (run once):
    python -m venv fraud_env
    fraud_env\Scripts\activate
    pip install pandas==2.1.4 numpy==1.26.4 scikit-learn==1.4.2
    pip install imbalanced-learn==0.12.3 matplotlib==3.8.4
    pip install openpyxl==3.1.2 pyinstaller==6.6.0

RUN:
    python build_exe.py

OUTPUT: dist\FraudDetector_UWS.exe

Evans Polley | B01823633 | UWS
"""
import subprocess, sys, os

HERE = os.path.dirname(os.path.abspath(__file__))
for f in ["app.py", "fraud_pipeline.py", "merged_fraud_dataset.csv"]:
    if not os.path.exists(os.path.join(HERE, f)):
        print(f"MISSING: {f}  — put all 3 files in the same folder first."); sys.exit(1)

cmd = [
    sys.executable, "-m", "PyInstaller",
    "--onefile", "--windowed", "--name", "FraudDetector_UWS", "--clean",
    f"--add-data=merged_fraud_dataset.csv{os.pathsep}.",
    f"--add-data=fraud_pipeline.py{os.pathsep}.",
    "--hidden-import=fraud_pipeline",
    "--hidden-import=sklearn.utils._cython_blas",
    "--hidden-import=sklearn.neighbors._partition_nodes",
    "--hidden-import=sklearn.tree._utils",
    "--hidden-import=sklearn.ensemble._forest",
    "--hidden-import=sklearn.linear_model._logistic",
    "--hidden-import=sklearn.svm._classes",
    "--hidden-import=sklearn.impute._base",
    "--hidden-import=sklearn.preprocessing._data",
    "--hidden-import=imblearn.over_sampling",
    "--hidden-import=imblearn.over_sampling._smote.base",
    "--hidden-import=matplotlib.backends.backend_tkagg",
    "--hidden-import=matplotlib.backends._backend_tk",
    "--hidden-import=matplotlib.backends.backend_agg",
    "--collect-all=matplotlib",
    "--collect-all=sklearn",
    "--collect-all=imblearn",
    "app.py"
]

print("Building FraudDetector_UWS.exe …")
result = subprocess.run(cmd, cwd=HERE)

if result.returncode == 0:
    exe = os.path.join(HERE, "dist", "FraudDetector_UWS.exe")
    sz  = os.path.getsize(exe)/1e6 if os.path.exists(exe) else 0
    print(f"\n✅  BUILD SUCCESSFUL — dist\\FraudDetector_UWS.exe  ({sz:.0f} MB)")
    print("    Bundled: merged_fraud_dataset.csv | fraud_pipeline.py")
    print("    Run:     dist\\FraudDetector_UWS.exe")
else:
    print("\n❌  BUILD FAILED")
    print("    1. Activate venv first:   fraud_env\\Scripts\\activate")
    print("    2. Confirm 64-bit Python: python -c \"import struct; print(struct.calcsize('P')*8,'bit')\"")
    print("    3. Reinstall:             pip install matplotlib==3.8.4 pyinstaller==6.6.0")
    sys.exit(1)
