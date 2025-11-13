# Fix ChromaDB Corruption Error

## Issue
If you see the error: "Could not connect to tenant default_tenant" or Rust panic errors, your ChromaDB database may be corrupted.

## Quick Fix

**Delete the corrupted database and let it recreate:**

```powershell
# From project root
Remove-Item -Recurse -Force .\artifacts\vector_store
```

Or manually delete the `artifacts/vector_store` folder.

## Automatic Fix

The code now automatically handles corruption by:
1. Detecting corruption/panic errors
2. Backing up the corrupted database
3. Recreating a fresh database
4. You'll need to re-index documents after this

## Re-index Documents

After fixing the database, run:

```powershell
python index_documents.py
```

Or in Docker:
```powershell
docker-compose run --rm cder-graphrag python index_documents.py
```

## Why This Happens

ChromaDB can get corrupted if:
- The process was interrupted during write operations
- Disk space ran out
- System crashed while ChromaDB was writing
- Version incompatibility

## Prevention

- Always shut down the application gracefully
- Ensure sufficient disk space
- Keep ChromaDB version consistent

