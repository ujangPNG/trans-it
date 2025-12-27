# Trans-It

Aplikasi rute TransJakarta termurah (Fare-Aware Routing).

- **Live Demo:** https://transit.ze4.me
- **Repository:** https://github.com/ujangPNG/trans-it

---

## Quick Start

**⚠️ Note:** Backend disarankan menggunakan **Linux** atau **WSL**.

### 1. Backend (Port 25200)

```bash
cd back
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
sh run.sh
```

### 2. Frontend (Port 3000)
```
cd front
npm install
cp .env.example .env.local
npm run dev
```
