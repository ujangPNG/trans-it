# chmod +x back/run.sh

cd ..
python3 -m py_compile back/app.py
uvicorn back.app:app --host 0.0.0.0 --port 25200