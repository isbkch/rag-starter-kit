# Database Migration: Metadata Column Rename

## Overview

This migration addresses a breaking schema change where the `metadata` columns in the SQLAlchemy models have been renamed to `metadata_` to avoid conflicts with Python's built-in `metadata` keyword and SQLAlchemy's naming conventions.

## Affected Tables

The following tables have had their `metadata` columns renamed to `metadata_`:

1. **documents** - Document metadata storage
2. **document_chunks** - Text chunk metadata  
3. **search_results** - Search result metadata

## Migration Details

- **Migration ID**: `002_rename_metadata_columns`
- **Revises**: `001_initial_schema`
- **Created**: 2024-12-19

## How to Apply the Migration

### Prerequisites

1. **Install PostgreSQL client libraries**:

   ```bash
   # macOS
   brew install postgresql
   
   # Ubuntu/Debian
   sudo apt-get install postgresql-dev
   
   # CentOS/RHEL
   sudo yum install postgresql-devel
   ```

2. **Install Python dependencies**:

   ```bash
   # Install dependencies using uv (recommended)
   uv sync

   # Or using pip
   pip install -r requirements.txt
   ```

3. **Update DATABASE_URL format** in `.env` file:

   ```bash
   # Correct format for psycopg (async driver)
   DATABASE_URL=postgresql+psycopg://user:password@host:port/database
   
   # NOT this (will cause psycopg2 import error):
   # DATABASE_URL=postgresql://user:password@host:port/database
   ```

### Running the Migration

1. **Validate migration (optional)**:

   ```bash
   cd backend/
   python validate_migration.py
   ```

2. **Check current migration status**:

   ```bash
   alembic current
   ```

3. **Test migration (dry run)**:

   ```bash
   alembic upgrade --sql head
   ```

4. **Apply the migration**:

   ```bash
   alembic upgrade head
   ```

5. **Verify the migration**:

   ```bash
   alembic history --verbose
   ```

### Rollback (if needed)

If you need to rollback this migration:

```bash
alembic downgrade 001_initial_schema
```

## Impact

- **Breaking Change**: Existing databases will encounter "column does not exist" errors without this migration
- **Data Safety**: This migration preserves all existing data, only renaming columns
- **Application Compatibility**: The SQLAlchemy models expect `metadata_` columns after this migration

## Testing

Before applying to production:

1. **Backup your database**
2. **Test in development environment**:

   ```bash
   # Run migration
   alembic upgrade head
   
   # Test application functionality
   pytest tests/
   ```

3. **Verify data integrity** after migration

## Troubleshooting

### Common Issues

1. **"alembic: command not found"**
   - Use `uv run alembic` instead of `alembic`
   - Or activate your virtual environment first

2. **Permission errors**
   - Ensure database user has ALTER TABLE permissions
   - Check database connection settings in `alembic.ini`

3. **Column already exists errors**
   - Check if migration was partially applied
   - Use `alembic current` to verify current state

### Manual SQL (Emergency Use Only)

If you need to apply the changes manually:

```sql
-- CAUTION: Backup your database first!

ALTER TABLE documents RENAME COLUMN metadata TO metadata_;
ALTER TABLE document_chunks RENAME COLUMN metadata TO metadata_;
ALTER TABLE search_results RENAME COLUMN metadata TO metadata_;
```

## Related Files

- Migration script: `alembic/versions/002_rename_metadata_columns.py`
- Model definitions: `app/models/database.py`
- Alembic config: `alembic.ini`
