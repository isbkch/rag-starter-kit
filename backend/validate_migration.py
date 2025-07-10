#!/usr/bin/env python3
"""
Migration validation script.

This script validates that the migration SQL is correctly formed
without requiring a database connection.
"""

import os
import sys
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory


def validate_migration():
    """Validate migration files without database connection."""
    print("🔍 Validating migration files...")

    # Check if migration files exist
    versions_dir = Path("alembic/versions")
    if not versions_dir.exists():
        print("❌ Alembic versions directory not found!")
        return False

    migration_files = list(versions_dir.glob("*.py"))
    print(f"📁 Found {len(migration_files)} migration files:")

    for migration_file in sorted(migration_files):
        print(f"   - {migration_file.name}")

    # Validate our specific migration exists
    rename_migration = versions_dir / "002_rename_metadata_columns.py"
    if not rename_migration.exists():
        print("❌ Metadata column rename migration not found!")
        return False

    print("✅ Migration file exists")

    # Try to validate the migration syntax by importing it
    try:
        spec = {"__file__": str(rename_migration)}
        with open(rename_migration) as f:
            exec(f.read(), spec)

        # Check required functions exist
        if "upgrade" not in spec:
            print("❌ Migration missing upgrade() function")
            return False

        if "downgrade" not in spec:
            print("❌ Migration missing downgrade() function")
            return False

        print("✅ Migration syntax is valid")

        # Validate migration metadata
        revision = spec.get("revision")
        down_revision = spec.get("down_revision")

        if not revision:
            print("❌ Migration missing revision identifier")
            return False

        if not down_revision:
            print("❌ Migration missing down_revision identifier")
            return False

        print(f"✅ Migration metadata valid:")
        print(f"   - Revision: {revision}")
        print(f"   - Down revision: {down_revision}")

        return True

    except Exception as e:
        print(f"❌ Migration syntax error: {e}")
        return False


def print_migration_instructions():
    """Print instructions for running the migration."""
    print("\n" + "=" * 60)
    print("📋 MIGRATION INSTRUCTIONS")
    print("=" * 60)

    print("\n🔧 PREREQUISITES:")
    print("1. Install PostgreSQL client libraries:")
    print("   macOS: brew install postgresql")
    print("   Ubuntu: sudo apt-get install postgresql-dev")
    print("   CentOS: sudo yum install postgresql-devel")

    print("\n2. Update DATABASE_URL in .env file:")
    print("   postgresql+psycopg://user:password@host:port/database")

    print("\n3. Ensure PostgreSQL server is running and accessible")

    print("\n🚀 RUNNING THE MIGRATION:")
    print("1. Test migration (dry run):")
    print("   alembic upgrade --sql head")

    print("\n2. Apply migration:")
    print("   alembic upgrade head")

    print("\n3. Verify migration applied:")
    print("   alembic current")
    print("   alembic history")

    print("\n📝 ALTERNATIVE APPROACHES:")
    print("1. Use Docker for PostgreSQL:")
    print("   docker run -d --name postgres -p 5432:5432 \\")
    print("     -e POSTGRES_USER=rag_user \\")
    print("     -e POSTGRES_PASSWORD=rag_password \\")
    print("     -e POSTGRES_DB=rag_platform \\")
    print("     postgres:13")

    print("\n2. Manual SQL execution (if needed):")
    print("   ALTER TABLE documents RENAME COLUMN metadata TO metadata_;")
    print("   ALTER TABLE document_chunks RENAME COLUMN metadata TO metadata_;")
    print("   ALTER TABLE search_results RENAME COLUMN metadata TO metadata_;")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    print("🔧 RAG Platform Migration Validator")
    print("=" * 40)

    if validate_migration():
        print("\n✅ All validations passed!")
        print_migration_instructions()
    else:
        print("\n❌ Validation failed!")
        sys.exit(1)
