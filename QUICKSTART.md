# Quick Start Guide

## Step 1: Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

## Step 2: Set Up Environment Variables

```bash
# Create .env file
python setup_env.py

# Edit .env with your credentials
# Required:
#   - NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
#   - OPENAI_API_KEY
```

### Getting Neo4j Credentials

**Option 1: Neo4j Aura (Cloud - Recommended)**
1. Go to https://neo4j.com/cloud/aura/
2. Create a free account
3. Create a new database instance
4. Copy the connection URI and password
5. Update `.env` with these values

**Option 2: Local Neo4j**
```bash
docker run -p 7687:7687 -p 7474:7474 \
  -e NEO4J_AUTH=neo4j/yourpassword \
  neo4j:latest
```
Then set in `.env`:
```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=yourpassword
```

### Getting OpenAI API Key

1. Go to https://platform.openai.com/api-keys
2. Create a new API key
3. Copy and paste into `.env` as `OPENAI_API_KEY`

## Step 3: Add Documents

Place your PDF or DOCX chapter files in:
```
data/cder_chapters/
```

Example:
```
data/cder_chapters/
  ├── chapter1.pdf
  ├── chapter2.pdf
  └── chapter3.docx
```

## Step 4: Run the System

### Interactive Mode (Recommended for first run)

```bash
python main.py
```

This starts an interactive chat where you can:
- Ask questions about your documents
- Compare different retrieval approaches
- View system statistics

### Single Query Mode

```bash
python main.py --mode query --query "What is parallel computing?" --retrieval hybrid
```

### Compare All Retrieval Approaches

```bash
python main.py --mode compare --query "Explain MapReduce"
```

## Step 5: Verify Setup

Check that everything is working:

```python
from src.chatbot import CDERChatbot

chatbot = CDERChatbot()
status = chatbot.get_system_status()
print(status)
```

## Troubleshooting

### "Neo4j connection failed"
- Verify your Neo4j URI, username, and password in `.env`
- For Aura, ensure URI starts with `neo4j+s://`
- For local, ensure Neo4j is running

### "OpenAI API key not found"
- Check that `OPENAI_API_KEY` is set in `.env`
- Verify the key is valid at https://platform.openai.com

### "No documents found"
- Ensure PDF/DOCX files are in `data/cder_chapters/`
- Check file permissions

### "Module not found" errors
- Ensure virtual environment is activated
- Run `pip install -r requirements.txt` again

## Next Steps

1. **Load Documents**: The system will process documents on first query
2. **Explore Notebooks**: Check `notebooks/` for interactive development
3. **Customize Config**: Edit `config/config.yaml` for your needs
4. **Run Tests**: `pytest tests/ -v` to verify installation

## Example Queries

Once documents are loaded, try:
- "What is the difference between parallel and distributed computing?"
- "Explain the MapReduce algorithm"
- "What are the benefits of message passing?"
- "Compare shared memory vs distributed memory architectures"

