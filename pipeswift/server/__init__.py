def create_app(*args, **kwargs):
    from .app import create_app as _create_app

    return _create_app(*args, **kwargs)


def run_server(*args, **kwargs):
    from .app import run_server as _run_server

    return _run_server(*args, **kwargs)


__all__ = ["create_app", "run_server"]
