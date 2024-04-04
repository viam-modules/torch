source .venv/bin/activate
pip3 install -r requirements.txt
python -m PyInstaller --onefile --hidden-import="googleapiclient" src/main.py