"""FastAPI surface: ingestion (R17), and later chat + glass-box (R19/R20).

The app is assembled in `app.create_app`; each feature area contributes a
router. Nothing here holds pipeline logic — routes translate HTTP into calls on
`domain` and `pedagogy`, and translate progress back into SSE.
"""

from socratic_tutor.api.app import create_app

__all__ = ["create_app"]
