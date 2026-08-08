"""Authentication routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from ..config import Settings
from ..deps import get_settings, get_templates

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
    settings: Settings = Depends(get_settings),
    templates: Jinja2Templates = Depends(get_templates),
):
    if username == settings.admin_username and password == settings.admin_password:
        request.session["authenticated"] = True
        request.session["username"] = username
        request.session["groups"] = list(settings.default_groups)
        return RedirectResponse(url=request.url_for("dashboard"), status_code=303)

    return templates.TemplateResponse(
        request, "pages/login.html", {"error": "Invalid credentials", "active_page": ""},
    )


@router.get("/logout", name="logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url=request.url_for("login_page"), status_code=303)
