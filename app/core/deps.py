"""deps.py — tiny accessors so controllers get shared services from app.state without importing globals."""
from fastapi import Request


def get_settings(request: Request):
    return request.app.state.settings


def get_jobs(request: Request):
    return request.app.state.jobs


def get_registry(request: Request):
    return request.app.state.registry
