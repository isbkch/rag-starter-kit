"""Rename metadata columns to metadata_

Revision ID: 002_rename_metadata_columns
Revises: 001_initial_schema
Create Date: 2024-12-19 11:24:00.000000

"""
import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "002_rename_metadata_columns"
down_revision = "001_initial_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Rename metadata columns to metadata_ to match SQLAlchemy model definitions."""

    # Rename metadata column to metadata_ in documents table
    op.alter_column(
        "documents",
        "metadata",
        new_column_name="metadata_",
        existing_type=sa.JSON(),
        existing_nullable=True,
    )

    # Rename metadata column to metadata_ in document_chunks table
    op.alter_column(
        "document_chunks",
        "metadata",
        new_column_name="metadata_",
        existing_type=sa.JSON(),
        existing_nullable=True,
    )

    # Rename metadata column to metadata_ in search_results table
    op.alter_column(
        "search_results",
        "metadata",
        new_column_name="metadata_",
        existing_type=sa.JSON(),
        existing_nullable=True,
    )


def downgrade() -> None:
    """Revert metadata_ columns back to metadata."""

    # Revert metadata_ column back to metadata in search_results table
    op.alter_column(
        "search_results",
        "metadata_",
        new_column_name="metadata",
        existing_type=sa.JSON(),
        existing_nullable=True,
    )

    # Revert metadata_ column back to metadata in document_chunks table
    op.alter_column(
        "document_chunks",
        "metadata_",
        new_column_name="metadata",
        existing_type=sa.JSON(),
        existing_nullable=True,
    )

    # Revert metadata_ column back to metadata in documents table
    op.alter_column(
        "documents",
        "metadata_",
        new_column_name="metadata",
        existing_type=sa.JSON(),
        existing_nullable=True,
    )
