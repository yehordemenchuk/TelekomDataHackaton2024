from datetime import datetime, timedelta
from typing import Optional, Union
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
import logging

from .config import settings
from .database import get_db
from .models.database import User
from .models.schemas import UserResponse

logger = logging.getLogger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT token scheme
security = HTTPBearer()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Generate password hash."""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)
    return encoded_jwt

def verify_token(token: str) -> Optional[dict]:
    """Verify and decode a JWT token."""
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
        return payload
    except JWTError as e:
        logger.warning(f"Token verification failed: {str(e)}")
        return None

def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
    """Authenticate a user with username and password."""
    user = db.query(User).filter(User.username == username).first()
    
    if not user:
        return None
    
    if not verify_password(password, user.hashed_password):
        return None
    
    if not user.is_active:
        return None
    
    return user

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """Get the current authenticated user from JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        token = credentials.credentials
        payload = verify_token(token)
        
        if payload is None:
            raise credentials_exception
        
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
            
    except JWTError:
        raise credentials_exception
    
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Inactive user"
        )
    
    return user

async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """Get the current active user."""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user

async def get_current_admin_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """Get the current user and verify admin privileges."""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return current_user

def create_user(db: Session, username: str, email: str, password: str, is_admin: bool = False) -> User:
    """Create a new user."""
    try:
        # Check if user already exists
        existing_user = db.query(User).filter(
            (User.username == username) | (User.email == email)
        ).first()
        
        if existing_user:
            raise ValueError("User with this username or email already exists")
        
        # Create new user
        hashed_password = get_password_hash(password)
        user = User(
            username=username,
            email=email,
            hashed_password=hashed_password,
            is_admin=is_admin,
            is_active=True
        )
        
        db.add(user)
        db.commit()
        db.refresh(user)
        
        logger.info(f"Created user: {username}")
        return user
        
    except Exception as e:
        logger.error(f"Error creating user {username}: {str(e)}")
        db.rollback()
        raise

def update_user_password(db: Session, user_id: int, new_password: str) -> bool:
    """Update a user's password."""
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return False
        
        user.hashed_password = get_password_hash(new_password)
        db.commit()
        
        logger.info(f"Updated password for user: {user.username}")
        return True
        
    except Exception as e:
        logger.error(f"Error updating password for user {user_id}: {str(e)}")
        db.rollback()
        return False

def deactivate_user(db: Session, user_id: int) -> bool:
    """Deactivate a user account."""
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return False
        
        user.is_active = False
        db.commit()
        
        logger.info(f"Deactivated user: {user.username}")
        return True
        
    except Exception as e:
        logger.error(f"Error deactivating user {user_id}: {str(e)}")
        db.rollback()
        return False

def activate_user(db: Session, user_id: int) -> bool:
    """Activate a user account."""
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return False
        
        user.is_active = True
        db.commit()
        
        logger.info(f"Activated user: {user.username}")
        return True
        
    except Exception as e:
        logger.error(f"Error activating user {user_id}: {str(e)}")
        db.rollback()
        return False

def get_user_by_username(db: Session, username: str) -> Optional[User]:
    """Get a user by username."""
    return db.query(User).filter(User.username == username).first()

def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """Get a user by email."""
    return db.query(User).filter(User.email == email).first()

def get_users(db: Session, skip: int = 0, limit: int = 100) -> list[User]:
    """Get a list of users with pagination."""
    return db.query(User).offset(skip).limit(limit).all()

class AuthenticationError(Exception):
    """Custom authentication error."""
    pass

class AuthorizationError(Exception):
    """Custom authorization error."""
    pass

def require_permission(permission: str):
    """Decorator to require specific permissions (placeholder for future role-based access)."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Placeholder for permission checking logic
            # In a full implementation, this would check user roles and permissions
            return await func(*args, **kwargs)
        return wrapper
    return decorator

# Rate limiting helpers (basic implementation)
from collections import defaultdict
import time

class RateLimiter:
    """Simple rate limiter for authentication attempts."""
    
    def __init__(self, max_attempts: int = 5, window_seconds: int = 300):
        self.max_attempts = max_attempts
        self.window_seconds = window_seconds
        self.attempts = defaultdict(list)
    
    def is_allowed(self, identifier: str) -> bool:
        """Check if an identifier is within rate limits."""
        now = time.time()
        
        # Clean old attempts
        self.attempts[identifier] = [
            attempt_time for attempt_time in self.attempts[identifier]
            if now - attempt_time < self.window_seconds
        ]
        
        # Check if under limit
        if len(self.attempts[identifier]) >= self.max_attempts:
            return False
        
        # Record this attempt
        self.attempts[identifier].append(now)
        return True
    
    def reset(self, identifier: str):
        """Reset rate limit for an identifier."""
        if identifier in self.attempts:
            del self.attempts[identifier]

# Global rate limiter instance
auth_rate_limiter = RateLimiter()

def check_auth_rate_limit(identifier: str) -> bool:
    """Check if authentication attempts are within rate limits."""
    return auth_rate_limiter.is_allowed(identifier)

def reset_auth_rate_limit(identifier: str):
    """Reset authentication rate limit for an identifier."""
    auth_rate_limiter.reset(identifier) 