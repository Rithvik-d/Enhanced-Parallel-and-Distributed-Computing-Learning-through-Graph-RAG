# Deployment Guide

Complete guide for deploying the CDER GraphRAG System.

## Table of Contents

1. [Docker Deployment](#docker-deployment)
2. [Production Considerations](#production-considerations)
3. [Environment Setup](#environment-setup)
4. [Monitoring](#monitoring)
5. [Scaling](#scaling)

---

## Docker Deployment

### Prerequisites

- Docker Desktop installed and running
- Docker Compose v2.0+
- At least 4GB RAM available
- 10GB free disk space

### Step-by-Step Deployment

#### 1. Clone and Setup

```bash
git clone <repository-url>
cd cder-graphrag
```

#### 2. Configure Environment

Create `.env` file:

```env
# Neo4j Configuration
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-secure-password
NEO4J_DATABASE=neo4j

# OpenAI Configuration
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-proj-your-api-key-here
```

#### 3. Build Docker Images

```bash
docker compose build
```

**Expected time**: 3-5 minutes

#### 4. Index Documents

```bash
docker compose run --rm cder-graphrag python index_documents.py
```

**Expected time**: 5-15 minutes (depends on document count)

#### 5. Start Services

```bash
# Start frontend
docker compose up frontend -d

# Or start all services
docker compose up -d
```

#### 6. Verify Deployment

```bash
# Check container status
docker compose ps

# Check logs
docker compose logs frontend --tail 50

# Test endpoint
curl http://localhost:8000
```

### Docker Commands Reference

```bash
# Build
docker compose build

# Start in background
docker compose up -d

# Start with logs
docker compose up

# Stop services
docker compose stop

# Stop and remove containers
docker compose down

# View logs
docker compose logs -f

# View specific service logs
docker compose logs frontend -f

# Rebuild without cache
docker compose build --no-cache

# Execute commands in container
docker compose run --rm cder-graphrag python main.py

# Access container shell
docker compose exec cder-graphrag /bin/bash
```

---

## Production Considerations

### Security

1. **Environment Variables**
   - Never commit `.env` file to version control
   - Use secrets management (Docker secrets, Kubernetes secrets, etc.)
   - Rotate API keys regularly

2. **Network Security**
   - Use HTTPS in production (reverse proxy: nginx, Traefik)
   - Restrict container network access
   - Use firewall rules

3. **API Keys**
   - Store in secure vault
   - Use least privilege principle
   - Monitor API usage

### Performance

1. **Resource Limits**

Add to `docker-compose.yml`:

```yaml
services:
  frontend:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

2. **Caching**

- Vector embeddings are cached in ChromaDB
- Graph data persists in Neo4j
- Consider Redis for session caching

3. **Rate Limiting**

Current settings:
- 2 seconds between requests
- Exponential backoff on errors
- Max 4 API calls per query

### High Availability

1. **Neo4j**
   - Use Neo4j Aura (managed, high availability)
   - Or deploy Neo4j cluster

2. **Application**
   - Use load balancer (nginx, HAProxy)
   - Deploy multiple frontend instances
   - Use shared volume for artifacts

3. **Monitoring**
   - Set up health checks
   - Monitor API usage
   - Track error rates

---

## Environment Setup

### Development

```bash
# Local development
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run locally
chainlit run frontend/app.py
```

### Staging

```bash
# Use Docker with staging config
docker compose -f docker-compose.staging.yml up
```

### Production

```bash
# Use production config
docker compose -f docker-compose.prod.yml up -d

# With resource limits
docker compose -f docker-compose.prod.yml up -d --scale frontend=3
```

---

## Monitoring

### Health Checks

Add health check to `docker-compose.yml`:

```yaml
services:
  frontend:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

### Logging

```bash
# View logs
docker compose logs -f

# Export logs
docker compose logs > logs.txt

# Log rotation (add to docker-compose.yml)
services:
  frontend:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

### Metrics

Monitor:
- API call count (should be 4 per query)
- Response times
- Error rates
- Token usage
- Neo4j connection status

---

## Scaling

### Horizontal Scaling

```bash
# Scale frontend service
docker compose up -d --scale frontend=3

# Use load balancer
# Configure nginx/traefik to distribute traffic
```

### Vertical Scaling

Increase resources in `docker-compose.yml`:

```yaml
services:
  frontend:
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
```

### Database Scaling

1. **Neo4j Aura**: Auto-scales
2. **ChromaDB**: Consider sharding for large datasets
3. **Caching**: Add Redis for frequently accessed data

---

## Backup and Recovery

### Backup

```bash
# Backup vector store
tar -czf vector_store_backup.tar.gz artifacts/vector_store/

# Backup Neo4j (if self-hosted)
neo4j-admin backup --backup-dir=/backups

# Backup configuration
cp .env .env.backup
cp config/config.yaml config/config.yaml.backup
```

### Recovery

```bash
# Restore vector store
tar -xzf vector_store_backup.tar.gz

# Rebuild if needed
docker compose build --no-cache
docker compose up -d
```

---

## Troubleshooting Deployment

### Container Won't Start

```bash
# Check logs
docker compose logs frontend

# Check resources
docker stats

# Restart
docker compose restart frontend
```

### Out of Memory

```bash
# Increase memory limit
# Edit docker-compose.yml
# Restart
docker compose up -d
```

### Port Already in Use

```bash
# Change port in docker-compose.yml
ports:
  - "8001:8000"  # Use different host port

# Or stop conflicting service
docker compose down
```

---

## Production Checklist

- [ ] Environment variables configured securely
- [ ] API keys stored securely (not in code)
- [ ] Documents indexed
- [ ] Health checks configured
- [ ] Logging configured
- [ ] Monitoring set up
- [ ] Backup strategy in place
- [ ] HTTPS configured (reverse proxy)
- [ ] Rate limiting configured
- [ ] Resource limits set
- [ ] Documentation updated

---

**Last Updated**: November 2025



