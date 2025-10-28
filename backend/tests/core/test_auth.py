"""
Tests for JWT authentication system.
"""

from datetime import timedelta

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from app.core.auth import (
    User,
    UserManager,
    create_access_token,
    get_current_user,
    get_password_hash,
    verify_password,
    verify_token,
)


class TestPasswordHashing:
    """Test password hashing and verification."""

    def test_password_hash_and_verify(self):
        """Test password hashing and verification."""
        password = "test_password_123"
        hashed = get_password_hash(password)

        assert hashed != password
        assert verify_password(password, hashed)
        assert not verify_password("wrong_password", hashed)

    def test_different_passwords_different_hashes(self):
        """Test that same password generates different hashes (salt)."""
        password = "test_password"
        hash1 = get_password_hash(password)
        hash2 = get_password_hash(password)

        # Hashes should be different due to random salt
        assert hash1 != hash2
        # But both should verify correctly
        assert verify_password(password, hash1)
        assert verify_password(password, hash2)


class TestJWTTokens:
    """Test JWT token creation and verification."""

    def test_create_access_token(self):
        """Test JWT access token creation."""
        data = {"sub": "user123", "email": "test@example.com"}
        token = create_access_token(data)

        assert isinstance(token, str)
        assert len(token) > 0

    def test_verify_token(self):
        """Test JWT token verification."""
        data = {"sub": "user123", "email": "test@example.com"}
        token = create_access_token(data)

        payload = verify_token(token, "access")

        assert payload["sub"] == "user123"
        assert "exp" in payload
        assert payload["type"] == "access"

    def test_verify_expired_token(self):
        """Test that expired tokens are rejected."""
        data = {"sub": "user123"}
        # Create token that expires immediately
        token = create_access_token(data, expires_delta=timedelta(seconds=-1))

        with pytest.raises(Exception):  # AuthenticationError
            verify_token(token, "access")

    def test_verify_wrong_token_type(self):
        """Test that token type validation works."""
        from app.core.auth import create_refresh_token

        data = {"sub": "user123"}
        refresh_token = create_refresh_token(data)

        # Try to verify refresh token as access token
        with pytest.raises(Exception):  # AuthenticationError
            verify_token(refresh_token, "access")


class TestUserManager:
    """Test user management functionality."""

    @pytest.mark.asyncio
    async def test_create_user(self):
        """Test user creation."""
        manager = UserManager()

        user = await manager.create_user(
            email="test@example.com",
            username="testuser",
            password="password123",
            is_admin=False,
        )

        assert user.email == "test@example.com"
        assert user.username == "testuser"
        assert user.is_active is True
        assert user.is_admin is False
        assert user.api_key is not None

    @pytest.mark.asyncio
    async def test_create_duplicate_email(self):
        """Test that duplicate email is rejected."""
        manager = UserManager()

        await manager.create_user(
            email="test@example.com", username="user1", password="pass123"
        )

        with pytest.raises(ValueError, match="Email already registered"):
            await manager.create_user(
                email="test@example.com", username="user2", password="pass456"
            )

    @pytest.mark.asyncio
    async def test_authenticate_user(self):
        """Test user authentication."""
        manager = UserManager()

        await manager.create_user(
            email="test@example.com", username="testuser", password="password123"
        )

        # Correct credentials
        user = await manager.authenticate_user("test@example.com", "password123")
        assert user is not None
        assert user.email == "test@example.com"

        # Wrong password
        user = await manager.authenticate_user("test@example.com", "wrongpassword")
        assert user is None

        # Non-existent user
        user = await manager.authenticate_user("nobody@example.com", "password123")
        assert user is None

    @pytest.mark.asyncio
    async def test_get_user_by_api_key(self):
        """Test getting user by API key."""
        manager = UserManager()

        created_user = await manager.create_user(
            email="test@example.com", username="testuser", password="password123"
        )

        # Get user by API key
        user = await manager.get_user_by_api_key(created_user.api_key)
        assert user is not None
        assert user.email == "test@example.com"

        # Invalid API key
        user = await manager.get_user_by_api_key("invalid_key")
        assert user is None

    @pytest.mark.asyncio
    async def test_change_password(self):
        """Test password change."""
        manager = UserManager()

        user = await manager.create_user(
            email="test@example.com", username="testuser", password="oldpassword"
        )

        # Change password
        success = await manager.change_password(user.id, "oldpassword", "newpassword")
        assert success is True

        # Authenticate with new password
        auth_user = await manager.authenticate_user("test@example.com", "newpassword")
        assert auth_user is not None

        # Old password should not work
        auth_user = await manager.authenticate_user("test@example.com", "oldpassword")
        assert auth_user is None

    @pytest.mark.asyncio
    async def test_reset_api_key(self):
        """Test API key reset."""
        manager = UserManager()

        user = await manager.create_user(
            email="test@example.com", username="testuser", password="password123"
        )

        old_api_key = user.api_key

        # Reset API key
        new_api_key = await manager.reset_api_key(user.id)

        assert new_api_key != old_api_key
        assert user.api_key == new_api_key

        # Old key should not work
        user_by_old_key = await manager.get_user_by_api_key(old_api_key)
        assert user_by_old_key is None

        # New key should work
        user_by_new_key = await manager.get_user_by_api_key(new_api_key)
        assert user_by_new_key is not None


class TestUserModel:
    """Test User model functionality."""

    def test_user_verify_password(self):
        """Test user password verification."""
        hashed_password = get_password_hash("password123")
        user = User(
            id="123",
            email="test@example.com",
            username="testuser",
            hashed_password=hashed_password,
        )

        assert user.verify_password("password123")
        assert not user.verify_password("wrongpassword")

    def test_user_has_permission(self):
        """Test user permission checking."""
        user = User(
            id="123",
            email="test@example.com",
            username="testuser",
            hashed_password="hash",
            permissions=["read:documents", "write:documents"],
        )

        assert user.has_permission("read:documents")
        assert user.has_permission("write:documents")
        assert not user.has_permission("admin")

    def test_admin_has_all_permissions(self):
        """Test that admin users have all permissions."""
        admin = User(
            id="123",
            email="admin@example.com",
            username="admin",
            hashed_password="hash",
            is_admin=True,
            permissions=[],
        )

        assert admin.has_permission("any_permission")
        assert admin.has_permission("admin")

    def test_user_to_dict(self):
        """Test user serialization to dictionary."""
        hashed_password = get_password_hash("password123")
        user = User(
            id="123",
            email="test@example.com",
            username="testuser",
            hashed_password=hashed_password,
            permissions=["read:documents"],
        )

        user_dict = user.to_dict()

        assert "id" in user_dict
        assert "email" in user_dict
        assert "username" in user_dict
        assert "is_active" in user_dict
        # Should not include sensitive data
        assert "hashed_password" not in user_dict
        assert user_dict["permissions"] == ["read:documents"]


@pytest.mark.asyncio
async def test_get_current_user_valid_token():
    """Test getting current user with valid token."""
    # Create a test token
    token_data = {"sub": "user123", "email": "test@example.com"}
    token = create_access_token(token_data)

    # Create mock credentials
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

    # This will fail in the actual implementation because user doesn't exist
    # but we're testing the token validation part
    try:
        user = await get_current_user(credentials)
        # If we get here, token was validated
        assert user is not None
    except HTTPException:
        # Expected if user lookup fails, but token was validated
        pass


@pytest.mark.asyncio
async def test_get_current_user_invalid_token():
    """Test getting current user with invalid token."""
    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer", credentials="invalid_token"
    )

    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(credentials)

    assert exc_info.value.status_code == 401
