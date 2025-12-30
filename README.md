# Trans-It

Aplikasi rute TransJakarta termurah (Fare-Aware Routing).

- **Live Demo:** https://transit.ze4.me
- **Repository:** https://github.com/ujangPNG/trans-it

---

## Quick Start

**⚠️ Note:** Backend disarankan menggunakan **Linux** atau **WSL**.

env.example udah di desain utk bisa langsung pake, jadi bisa langsung rename aja, atau tinggal copas semua command di bawah ini

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
```bash
cd front
npm install
cp .env.example .env.local
npm run dev
```
