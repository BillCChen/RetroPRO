molstar = None
try:
    from .molstar import molstar
except Exception:
    try:
        import sys
        sys.path.append("/home/chenqixuan/retro_star/retro_star/alg")
        from alg.molstar import molstar
    except Exception:
        molstar = None

try:
    from .molstar_parallel import molstar_parallel
except Exception:
    # Keep optional for backward compatibility in environments
    # where the parallel module dependencies are unavailable.
    molstar_parallel = None
