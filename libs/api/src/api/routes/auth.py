"""Authentication routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from ..deps import get_templates, get_user_manager
from ..users import UserManager

router = APIRouter(tags=["auth"])


@router.get("/login", name="login_page")
def login_page(
    request: Request,
    templates: Jinja2Templates = Depends(get_templates),
):
    if request.session.get("authenticated"):
        return RedirectResponse(url=request.url_for("dashboard"), status_code=303)
    return templates.TemplateResponse(
        request, "pages/login.html", {"error": None, "active_page": ""},
    )


@router.post("/login", name="login")
def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    user_mgr: UserManager = Depends(get_user_manager),
    templates: Jinja2Templates = Depends(get_templates),
):
    user = user_mgr.authenticate(username, password)
    if user is not None:
        request.session["authenticated"] = True
        request.session["username"] = user.username
        request.session["groups"] = user.groups
        request.session["is_admin"] = user.is_admin
        return RedirectResponse(url=request.url_for("dashboard"), status_code=303)

    return templates.TemplateResponse(
        request, "pages/login.html", {"error": "Invalid credentials", "active_page": ""},
    )


@router.get("/logout", name="logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url=request.url_for("login_page"), status_code=303)
