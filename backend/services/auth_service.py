"""
backend/services/auth_service.py
=================================
Admin authentication business logic (login, register).
"""

from fastapi import HTTPException
from jose import jwt
from sqlalchemy.orm import Session

from backend.config import SECRET_KEY, ALGORITHM, logger
from backend.core.security import verify_password, get_password_hash
from backend.models import AdminUser


def login_admin(username: str, password: str, db: Session) -> dict:
    """Verify credentials and return a JWT access token."""
    user = db.query(AdminUser).filter(AdminUser.username == username).first()
    if not user or not verify_password(password, user.hashed_password):
        logger.warning("Failed login attempt for user '%s'", username)
        raise HTTPException(status_code=400, detail="Incorrect username or password")

    access_token = jwt.encode({"sub": user.username}, SECRET_KEY, algorithm=ALGORITHM)
    logger.info("User '%s' logged in successfully", username)
    return {"access_token": access_token, "token_type": "bearer"}


def register_admin(username: str, password: str, setup_token: str, db: Session) -> dict:
    """Register a new admin user (protected by setup_token)."""
    if setup_token != SECRET_KEY:
        raise HTTPException(status_code=403, detail="Invalid setup token")

    existing_user = db.query(AdminUser).filter(AdminUser.username == username).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already registered")

    hashed_pw = get_password_hash(password)
    new_admin = AdminUser(username=username, hashed_password=hashed_pw)
    db.add(new_admin)
    db.commit()
    db.refresh(new_admin)
    logger.info("Registered new admin: '%s'", username)
    return {"msg": "Admin created successfully"}
