"""
How to mount this into the existing backend/main.py (2 lines):

    from backend.pullback.api import router as pullback_router
    app.include_router(pullback_router)

Nothing else in main.py needs to change - this router owns its own prefix
(/api/pullback/*), its own job state, and its own cache file
(data/last_result_pullback.json), so it can't collide with the primary/broad/quick/
newflow job state or cache files.
"""
