"""
backend/core/security.py
========================
Password hashing, JWT helpers.
"""

import bcrypt
from jose import JWTError, jwt
from fastapi import HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from backend.config import SECRET_KEY, ALGORITHM
from backend.core.dependencies import get_db

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
    except Exception:
        return False


def get_password_hash(password: str) -> str:
    pwd_bytes = password.encode("utf-8")
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(pwd_bytes, salt)
    return hashed_password.decode("utf-8")


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> str:
    """FastAPI dependency — validates JWT and returns username."""
    from backend.models import AdminUser  # local import avoids circular deps

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")

        user = db.query(AdminUser).filter(AdminUser.username == username).first()
        if not user:
            raise HTTPException(status_code=401, detail="User no longer exists. Please log in again.")

        return username
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
