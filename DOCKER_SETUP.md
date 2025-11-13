# Docker Setup for CDER GraphRAG (Neo4j Will Work!)

## Why Docker?

Docker runs your code in a Linux container where Neo4j connection works perfectly (just like Colab)!

## Prerequisites

1. **Install Docker Desktop for Windows**
   - Download from: https://www.docker.com/products/docker-desktop/
   - Install and restart your computer
   - Make sure Docker Desktop is running

## Quick Start

### Step 1: Build Docker Image

Open PowerShell in your project directory:

```powershell
cd "C:\Users\rithv\OneDrive\Desktop\New folder"
docker-compose build
```

Wait 3-5 minutes for build to complete.

### Step 2: Run Container

```powershell
docker-compose run --rm cder-graphrag
```

This opens a bash shell inside the container.

### Step 3: Test Neo4j

Inside the container:

```bash
python test_neo4j_direct.py
```

**Expected**: ✅ Connection successful!

### Step 4: Run Your System

```bash
python main.py
```

## Alternative: Run Directly

Instead of interactive shell, run directly:

```powershell
docker-compose run --rm cder-graphrag python main.py
```

Or test Neo4j:

```powershell
docker-compose run --rm cder-graphrag python test_neo4j_direct.py
```

## Quick Commands

### Build image:
```powershell
docker-compose build
```

### Run interactive shell:
```powershell
docker-compose run --rm cder-graphrag
```

### Run specific script:
```powershell
docker-compose run --rm cder-graphrag python main.py
docker-compose run --rm cder-graphrag python test_neo4j_direct.py
docker-compose run --rm cder-graphrag python test_query.py
```

### Stop and remove:
```powershell
docker-compose down
```

## Benefits

✅ Neo4j works (Linux environment)
✅ No WSL/Ubuntu setup needed
✅ Isolated environment
✅ Easy to share and deploy
✅ Works exactly like Colab

## Troubleshooting

### Docker not running
- Make sure Docker Desktop is started
- Check system tray for Docker icon

### Build fails
```powershell
docker-compose build --no-cache
```

### Permission issues
- Make sure Docker Desktop has access to your drive
- Settings → Resources → File Sharing → Add your drive

## That's It!

Docker gives you a Linux environment where Neo4j works, without needing WSL!

