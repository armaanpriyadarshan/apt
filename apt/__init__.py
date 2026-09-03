import importlib.metadata

try:
    __version__ = importlib.metadata.version("apt")
except importlib.metadata.PackageNotFoundError:
    pass
