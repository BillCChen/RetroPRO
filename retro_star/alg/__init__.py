molstar = None
try:
    from .molstar import molstar
except Exception:
    try:
        import os
        import sys
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from alg.molstar import molstar
    except Exception:
        molstar = None

try:
    from .molstar_parallel import molstar_parallel
except Exception:
    # Keep optional for backward compatibility in environments
    # where the parallel module dependencies are unavailable.
    molstar_parallel = None
