from functools import lru_cache

from jinja2 import Environment, PackageLoader


@lru_cache(maxsize=1)
def get_environment() -> Environment:
    return Environment(loader=PackageLoader("agency", "templates"))
