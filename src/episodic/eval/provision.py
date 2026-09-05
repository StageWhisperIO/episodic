import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


def cache_root():
    return Path(os.environ.get("EPISODIC_PROVISION_DIR",
                               os.path.expanduser("~/.cache/episodic-provision")))


def _run(args, cwd=None, timeout=1800):
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=timeout)


def _slug(repo, version):
    key = f"{repo}@{version}".encode()
    return repo.replace("/", "__") + "_" + hashlib.sha1(key).hexdigest()[:8]


def _pyenv_root():
    return Path(os.environ.get("PYENV_ROOT", os.path.expanduser("~/.pyenv")))


def _version_sort_key(name):
    parts = []
    for chunk in name.split("."):
        try:
            parts.append(int(chunk))
        except ValueError:
            parts.append(-1)
    return tuple(parts)


def _pyenv_versions():
    versions_dir = _pyenv_root() / "versions"
    if not versions_dir.is_dir():
        return []
    names = [p.name for p in versions_dir.iterdir() if p.is_dir()]
    return sorted(names, key=_version_sort_key, reverse=True)


def _matching_pyenv_python(python_version):
    for name in _pyenv_versions():
        if name == python_version or name.startswith(python_version + "."):
            candidate = _pyenv_root() / "versions" / name / "bin" / "python"
            if candidate.exists():
                return str(candidate)
    return None


def _resolve_interpreter(python_version):
    if not python_version:
        return sys.executable, None
    matched = _matching_pyenv_python(python_version)
    if matched:
        return matched, None
    which = shutil.which(f"python{python_version}")
    if which:
        return which, None
    available = ", ".join(_pyenv_versions()) or "none"
    mismatch = (f"requested python {python_version} not found among pyenv versions "
                f"({available}); falling back to {sys.executable}")
    return sys.executable, mismatch


def _interpreter(python_version):
    interpreter, _ = _resolve_interpreter(python_version)
    return interpreter


_SYSTEM_TOOLS = {"apt-get", "apt", "sudo", "yum", "dnf", "dpkg", "apk", "brew", "add-apt-repository"}


def _pre_install_commands(install_config):
    commands = []
    for step in (install_config.get("pre_install") or []):
        parts = shlex.split(step)
        if parts and parts[0] not in _SYSTEM_TOOLS:
            commands.append(parts)
    return commands


def _pip_commands(install_config, venv_python, no_deps=False):
    pip = [venv_python, "-m", "pip", "install", "--disable-pip-version-check", "-q"]
    install = install_config.get("install") or "pip install -e ."
    install = install.replace("pip install -e", "pip install").replace("pip install", "").strip()
    install_args = shlex.split(install)
    if no_deps:
        install_args = ["--no-deps"] + install_args
    commands = [pip + install_args]
    pip_packages = install_config.get("pip_packages") or []
    for package in pip_packages:
        commands.append(pip + [package])
    if not any("pytest" in package for package in pip_packages):
        commands.append(pip + ["pytest"])
    return commands


def _clean_frozen_requirements(text):
    lines = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("--"):
            continue
        if line.startswith("-e ") or line.startswith("-r "):
            continue
        if "file://" in line or line.startswith(("git+", "hg+", "svn+", "bzr+")):
            continue
        lines.append(line)
    return "\n".join(lines)


def _install_pinned_requirements(base, checkout, venv_python, requirements_text):
    cleaned = _clean_frozen_requirements(requirements_text)
    if not cleaned:
        return False
    frozen_path = base / "frozen-requirements.txt"
    frozen_path.write_text(cleaned)
    try:
        result = _run([venv_python, "-m", "pip", "install", "--disable-pip-version-check", "-q",
                        "--no-deps", "-r", str(frozen_path)], cwd=str(checkout), timeout=1800)
    except Exception as exc:
        (base / "pinned-install.log").write_text(f"{type(exc).__name__}: {exc}")
        return False
    if result.returncode != 0:
        (base / "pinned-install.log").write_text((result.stdout + result.stderr)[-4000:])
        return False
    return True


def ensure_repo_venv(instance, force=False):
    repo = instance.get("repo") or ""
    version = str(instance.get("version") or "")
    install_config = instance.get("install_config") or {}
    setup_commit = instance.get("environment_setup_commit") or instance.get("base_commit")
    if not (repo and setup_commit):
        return None

    base = cache_root() / _slug(repo, version)
    venv = base / "venv"
    venv_python = str(venv / "bin" / "python")
    ok_marker = base / ".provisioned"
    fail_marker = base / ".failed"

    if not force and ok_marker.exists():
        return {"python": venv_python, "test_env": _env_from(base, install_config),
                "provision_notes": _load_provision_notes(base)}
    if not force and fail_marker.exists():
        return None

    base.mkdir(parents=True, exist_ok=True)
    checkout = base / "checkout"
    remote = f"https://github.com/{repo}.git"
    notes = {}
    try:
        if not (checkout / ".git").exists():
            _, code = _clone(remote, checkout)
            if code != 0:
                return _fail(base, fail_marker)
            _run(["git", "-C", str(checkout), "checkout", setup_commit], timeout=120)

        interpreter, mismatch = _resolve_interpreter(install_config.get("python"))
        if mismatch:
            notes["interpreter_mismatch"] = mismatch
            (base / "interpreter-mismatch.log").write_text(mismatch)

        if _run([interpreter, "-m", "venv", str(venv)], timeout=300).returncode != 0:
            return _fail(base, fail_marker)
        _run([venv_python, "-m", "pip", "install", "--disable-pip-version-check", "-q",
              "-U", "pip", "setuptools", "wheel"], timeout=600)
        for command in _pre_install_commands(install_config):
            _run(command, cwd=str(checkout), timeout=600)

        pinned = _install_pinned_requirements(base, checkout, venv_python, instance.get("requirements"))
        notes["pinned_requirements"] = pinned

        for command in _pip_commands(install_config, venv_python, no_deps=pinned):
            result = _run(command, cwd=str(checkout), timeout=1800)
            if result.returncode != 0:
                (base / "install.log").write_text((result.stdout + result.stderr)[-4000:])
                return _fail(base, fail_marker)
    except Exception as exc:
        (base / "install.log").write_text(str(exc))
        return _fail(base, fail_marker)

    shutil.rmtree(checkout, ignore_errors=True)
    ok_marker.write_text("ok")
    (base / "provision-notes.json").write_text(json.dumps(notes))
    return {"python": venv_python, "test_env": _env_from(base, install_config), "provision_notes": notes}


def _clone(remote, dest):
    _, code = _run_clone(["git", "clone", "--filter=blob:none", "--quiet", remote, str(dest)])
    if code != 0:
        shutil.rmtree(dest, ignore_errors=True)
        _, code = _run_clone(["git", "clone", "--quiet", remote, str(dest)])
    return None, code


def _run_clone(args):
    result = _run(args, timeout=600)
    return result.stdout + result.stderr, result.returncode


def _env_from(base, install_config):
    venv_bin = str(base / "venv" / "bin")
    env = {"PATH": venv_bin + os.pathsep + os.environ.get("PATH", "")}
    for key, value in (install_config.get("env_vars") or {}).items():
        env[str(key)] = str(value)
    return env


def _load_provision_notes(base):
    path = base / "provision-notes.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _fail(base, fail_marker):
    fail_marker.write_text("failed")
    return None


def provision_episode(episode, instance, force=False):
    result = ensure_repo_venv(instance, force=force)
    if result is None:
        return False
    episode["repo_state"]["test_env"] = result["test_env"]
    if result.get("provision_notes"):
        episode["repo_state"]["provision_notes"] = result["provision_notes"]
    return True
