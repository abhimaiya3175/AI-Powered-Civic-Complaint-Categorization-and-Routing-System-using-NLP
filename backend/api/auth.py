"""
backend/api/auth.py
===================
Authentication routes: /login and /register-admin.
"""

from fastapi import APIRouter, Depends
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from backend.core.dependencies import get_db
from backend.schemas import AdminCreate
from backend.services.auth_service import login_admin, register_admin

router = APIRouter()


@router.post("/login")
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    """Authenticate admin user and return JWT token."""
    return login_admin(form_data.username, form_data.password, db)


@router.post("/register-admin")
def register_admin_endpoint(
    admin: AdminCreate,
    db: Session = Depends(get_db),
):
    """Register a new admin user. Requires setup_token for security."""
    return register_admin(admin.username, admin.password, admin.setup_token, db)
