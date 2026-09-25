"""run.py - start the web app:  python run.py   (or: uvicorn app.main:create_app --factory)"""
import uvicorn

from app.config import settings

if __name__ == "__main__":
    uvicorn.run("app.main:create_app", factory=True, host=settings.host, port=settings.port, reload=settings.debug)
