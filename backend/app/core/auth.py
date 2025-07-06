"""
Authentication and authorization system using JWT tokens.
"""

import logging
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from passlib.context import CryptContext
from jose import JWTError, jwt
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.core.config import settings

logger = logging.getLogger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT token security
security = HTTPBearer()

# Token types
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7


class AuthenticationError(Exception):
    """Authentication related errors."""
    pass


class AuthorizationError(Exception):
    """Authorization related errors."""
    pass


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire, "type": "access"})
    
    try:
        encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm="HS256")
        return encoded_jwt
    except Exception as e:
        logger.error(f"Error creating access token: {e}")
        raise AuthenticationError("Failed to create access token")


def create_refresh_token(data: Dict[str, Any]) -> str:
    """Create JWT refresh token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    
    try:
        encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm="HS256")
        return encoded_jwt
    except Exception as e:
        logger.error(f"Error creating refresh token: {e}")
        raise AuthenticationError("Failed to create refresh token")


def verify_token(token: str, token_type: str = "access") -> Dict[str, Any]:
    """Verify and decode JWT token."""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
        
        # Check token type
        if payload.get("type") != token_type:
            raise AuthenticationError(f"Invalid token type. Expected {token_type}")
        
        # Check expiration
        exp = payload.get("exp")
        if exp is None:
            raise AuthenticationError("Token missing expiration")
        
        if datetime.utcnow() > datetime.utcfromtimestamp(exp):
            raise AuthenticationError("Token has expired")
        
        return payload
        
    except JWTError as e:
        logger.warning(f"JWT verification failed: {e}")
        raise AuthenticationError("Invalid token")
    except Exception as e:
        logger.error(f"Token verification error: {e}")
        raise AuthenticationError("Token verification failed")


def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def generate_api_key() -> str:
    """Generate a secure API key."""
    return secrets.token_urlsafe(32)


class User:
    """User model for authentication."""
    
    def __init__(
        self,
        id: str,
        email: str,
        username: str,
        hashed_password: str,
        is_active: bool = True,
        is_admin: bool = False,
        api_key: Optional[str] = None,
        created_at: Optional[datetime] = None,
        last_login: Optional[datetime] = None,
        permissions: Optional[List[str]] = None
    ):
        self.id = id
        self.email = email
        self.username = username
        self.hashed_password = hashed_password
        self.is_active = is_active
        self.is_admin = is_admin
        self.api_key = api_key
        self.created_at = created_at or datetime.utcnow()
        self.last_login = last_login
        self.permissions = permissions or []
    
    def verify_password(self, password: str) -> bool:
        """Verify user password."""
        return verify_password(password, self.hashed_password)
    
    def has_permission(self, permission: str) -> bool:
        """Check if user has specific permission."""
        if self.is_admin:
            return True
        return permission in self.permissions
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert user to dictionary (excluding sensitive data)."""
        return {
            "id": self.id,
            "email": self.email,
            "username": self.username,
            "is_active": self.is_active,
            "is_admin": self.is_admin,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "permissions": self.permissions
        }


class UserManager:
    """User management operations."""
    
    def __init__(self):
        # In production, this would connect to a database
        self._users: Dict[str, User] = {}
        self._email_to_id: Dict[str, str] = {}
        self._username_to_id: Dict[str, str] = {}
        self._api_keys: Dict[str, str] = {}  # api_key -> user_id
    
    async def create_user(
        self,
        email: str,
        username: str,
        password: str,
        is_admin: bool = False,
        permissions: Optional[List[str]] = None
    ) -> User:
        """Create a new user."""
        # Check if email or username already exists
        if email in self._email_to_id:
            raise ValueError("Email already registered")
        
        if username in self._username_to_id:
            raise ValueError("Username already taken")
        
        # Generate user ID and hash password
        user_id = secrets.token_hex(16)
        hashed_password = get_password_hash(password)
        api_key = generate_api_key()
        
        # Create user
        user = User(
            id=user_id,
            email=email,
            username=username,
            hashed_password=hashed_password,
            is_admin=is_admin,
            api_key=api_key,
            permissions=permissions or []
        )
        
        # Store user
        self._users[user_id] = user
        self._email_to_id[email] = user_id
        self._username_to_id[username] = user_id
        self._api_keys[api_key] = user_id
        
        logger.info(f"Created new user: {username} ({email})")
        return user
    
    async def authenticate_user(self, email: str, password: str) -> Optional[User]:
        """Authenticate user with email and password."""
        user_id = self._email_to_id.get(email)
        if not user_id:
            return None
        
        user = self._users.get(user_id)
        if not user or not user.is_active:
            return None
        
        if not user.verify_password(password):
            return None
        
        # Update last login
        user.last_login = datetime.utcnow()
        
        logger.info(f"User authenticated: {user.username}")
        return user
    
    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return self._users.get(user_id)
    
    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        user_id = self._email_to_id.get(email)
        return self._users.get(user_id) if user_id else None
    
    async def get_user_by_api_key(self, api_key: str) -> Optional[User]:
        """Get user by API key."""
        user_id = self._api_keys.get(api_key)
        return self._users.get(user_id) if user_id else None
    
    async def update_user(self, user_id: str, **kwargs) -> Optional[User]:
        """Update user information."""
        user = self._users.get(user_id)
        if not user:
            return None
        
        # Update allowed fields
        allowed_fields = ["email", "username", "is_active", "is_admin", "permissions"]
        
        for field, value in kwargs.items():
            if field in allowed_fields:
                setattr(user, field, value)
        
        return user
    
    async def change_password(self, user_id: str, old_password: str, new_password: str) -> bool:
        """Change user password."""
        user = self._users.get(user_id)
        if not user:
            return False
        
        if not user.verify_password(old_password):
            return False
        
        user.hashed_password = get_password_hash(new_password)
        logger.info(f"Password changed for user: {user.username}")
        return True
    
    async def reset_api_key(self, user_id: str) -> Optional[str]:
        """Reset user API key."""
        user = self._users.get(user_id)
        if not user:
            return None
        
        # Remove old API key
        if user.api_key and user.api_key in self._api_keys:
            del self._api_keys[user.api_key]
        
        # Generate new API key
        new_api_key = generate_api_key()
        user.api_key = new_api_key
        self._api_keys[new_api_key] = user_id
        
        logger.info(f"API key reset for user: {user.username}")
        return new_api_key
    
    async def delete_user(self, user_id: str) -> bool:
        """Delete user."""
        user = self._users.get(user_id)
        if not user:
            return False
        
        # Remove from all indexes
        del self._users[user_id]
        del self._email_to_id[user.email]
        del self._username_to_id[user.username]
        
        if user.api_key and user.api_key in self._api_keys:
            del self._api_keys[user.api_key]
        
        logger.info(f"Deleted user: {user.username}")
        return True


# Global user manager instance
user_manager = UserManager()


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """Get current authenticated user from JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Try JWT token first
        payload = verify_token(credentials.credentials, "access")
        user_id = payload.get("sub")
        
        if user_id is None:
            raise credentials_exception
        
        user = await user_manager.get_user_by_id(user_id)
        if user is None or not user.is_active:
            raise credentials_exception
        
        return user
        
    except AuthenticationError:
        # Try API key as fallback
        user = await user_manager.get_user_by_api_key(credentials.credentials)
        if user is None or not user.is_active:
            raise credentials_exception
        
        return user


async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Get current active user."""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user


async def get_current_admin_user(current_user: User = Depends(get_current_active_user)) -> User:
    """Get current admin user."""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return current_user


def require_permission(permission: str):
    """Decorator to require specific permission."""
    def permission_checker(current_user: User = Depends(get_current_active_user)) -> User:
        if not current_user.has_permission(permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{permission}' required"
            )
        return current_user
    
    return permission_checker


# Permission constants
class Permissions:
    """Permission constants."""
    READ_DOCUMENTS = "read:documents"
    WRITE_DOCUMENTS = "write:documents"
    DELETE_DOCUMENTS = "delete:documents"
    SEARCH = "search"
    UPLOAD = "upload"
    ADMIN = "admin"
    MANAGE_USERS = "manage:users"
    VIEW_ANALYTICS = "view:analytics"


# Authentication utilities
async def login_user(email: str, password: str) -> Dict[str, Any]:
    """Login user and return tokens."""
    user = await user_manager.authenticate_user(email, password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )
    
    # Create tokens
    access_token = create_access_token(data={"sub": user.id})
    refresh_token = create_refresh_token(data={"sub": user.id})
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "user": user.to_dict()
    }


async def refresh_access_token(refresh_token: str) -> Dict[str, Any]:
    """Refresh access token using refresh token."""
    try:
        payload = verify_token(refresh_token, "refresh")
        user_id = payload.get("sub")
        
        if user_id is None:
            raise AuthenticationError("Invalid refresh token")
        
        user = await user_manager.get_user_by_id(user_id)
        if user is None or not user.is_active:
            raise AuthenticationError("User not found or inactive")
        
        # Create new access token
        access_token = create_access_token(data={"sub": user.id})
        
        return {
            "access_token": access_token,
            "token_type": "bearer"
        }
        
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e)
        )


# Initialize default admin user
async def create_default_admin():
    """Create default admin user if none exists."""
    try:
        admin_email = "admin@rag-platform.com"
        admin_password = settings.DEFAULT_ADMIN_PASSWORD or "admin123"
        
        # Check if admin already exists
        existing_admin = await user_manager.get_user_by_email(admin_email)
        if existing_admin:
            logger.info("Default admin user already exists")
            return
        
        # Create admin user
        admin_user = await user_manager.create_user(
            email=admin_email,
            username="admin",
            password=admin_password,
            is_admin=True,
            permissions=[p for p in dir(Permissions) if not p.startswith('_')]
        )
        
        logger.info(f"Created default admin user: {admin_email}")
        logger.warning(f"Default admin password: {admin_password}")
        logger.warning("Please change the default admin password immediately!")
        
    except Exception as e:
        logger.error(f"Failed to create default admin user: {e}")


# Rate limiting for auth endpoints
AUTH_RATE_LIMITS = {
    "login": "5/minute",
    "register": "3/minute", 
    "refresh": "10/minute",
    "reset_password": "3/minute"
}
