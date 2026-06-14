import hashlib

from ..schema import short_id


def new_id(prefix):
    return f"{prefix}_{short_id()}"


def episode_id_from_session(session_id):
    digest = hashlib.sha1(session_id.encode("utf-8")).hexdigest()[:12]
    return f"ep_{digest}"
