# 6. Deployment & Infrastructure Guide

This document provides a guide for deploying and managing the infrastructure of the Enterprise RAG Platform. It is intended for DevOps engineers and system administrators.

## Production Environment

For a production setup, it is recommended to use the `docker-compose.prod.yml` file, which is optimized for production use. It is also advisable to use a managed cloud provider for services like PostgreSQL, Redis, and Elasticsearch for better scalability, reliability, and maintainability.

### Environment Variables

Create a `.env.production` file based on the `.env.example`. Key variables to configure for production include:

* `SECRET_KEY`: A strong, randomly generated secret key for cryptographic signing.
* `DATABASE_URL`: The connection string for your production PostgreSQL database.
* `REDIS_URL`: The connection string for your production Redis instance.
* `VECTOR_DB_PROVIDER`: The selected vector database provider (`chroma`, `pinecone`, or `weaviate`).
* `OPENAI_API_KEY`: Your OpenAI API key.
* Provider-specific API keys and environment settings for Pinecone or Weaviate if used.

### Running in Production Mode

To start the services in production mode, use the following command:

```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

## Scalability

The platform is designed to be scalable. Here are the primary components to consider for scaling:

* **Backend API**: The FastAPI application is stateless and can be scaled horizontally by running multiple containers behind a load balancer.
* **Celery Workers**: The number of Celery workers can be increased to handle a higher volume of document ingestion tasks. You can scale the `celery-worker` service using Docker Compose:

    ```bash
    docker-compose up -d --scale celery-worker=4
    ```

* **Databases**: For high-traffic environments, consider using managed database services (e.g., Amazon RDS for PostgreSQL, ElastiCache for Redis) that support read replicas and clustering.

## Database Migrations

The platform uses Alembic for managing database schema migrations. Migrations are essential for updating the database schema in a controlled and versioned manner.

* **Generating a New Migration**:
    After making changes to the SQLAlchemy models in `app/models/`, generate a new migration script:

    ```bash
    cd backend && uv run alembic revision --autogenerate -m "Your migration description"
    ```

* **Applying Migrations**:
    To apply all pending migrations to the database, run:

    ```bash
    cd backend && uv run alembic upgrade head
    ```

    This command should be run as part of your deployment process.

## Backup and Recovery

* **PostgreSQL**: Use standard database backup tools like `pg_dump` to create regular backups of your document metadata and application data.
* **Vector Database**: Follow the backup and recovery procedures recommended by your chosen vector database provider.
* **Elasticsearch**: Use Elasticsearch snapshots to back up your keyword search indexes.

## Monitoring

The platform includes a monitoring stack with Prometheus and Grafana. The Grafana dashboards are pre-configured to provide an overview of the system's health and performance. Ensure that the monitoring services are running and accessible in your production environment.
