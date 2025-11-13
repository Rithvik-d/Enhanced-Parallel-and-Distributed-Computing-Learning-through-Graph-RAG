# 🚀 Running the Frontend

## ✅ Frontend is Starting!

The Chainlit frontend is now running. Here's how to access it:

## 🌐 Access the Web Interface

**Open your browser and go to:**
```
http://localhost:8000
```

## 📋 What You'll See

1. **Welcome Screen** - Introduction to the CDER GraphRAG system
2. **Chat Interface** - Enter your questions here
3. **4 Responses** - Each query shows:
   - No-RAG (Baseline)
   - Vector-Only RAG
   - Graph-Only RAG
   - Hybrid RAG

## 🎯 Try These Questions

- "What is parallel computing?"
- "Explain MapReduce architecture"
- "What are the benefits of distributed systems?"
- "How does load balancing work?"

## 🔧 If Frontend Doesn't Start

### Option 1: Check if it's running
```powershell
netstat -an | findstr ":8000"
```

### Option 2: Run manually
```powershell
cd "c:\Users\rithv\OneDrive\Desktop\New folder"
chainlit run frontend/app.py --port 8000
```

### Option 3: Use Docker
```powershell
docker-compose up frontend
```

## 📊 Features

- ✅ All 4 retrieval modes displayed simultaneously
- ✅ Performance metrics for each mode
- ✅ Citations and source information
- ✅ Comparison table
- ✅ Real-time processing

## 🛑 To Stop

Press `Ctrl+C` in the terminal where Chainlit is running.

