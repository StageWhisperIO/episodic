import hashlib
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


def _interpreter(python_version):
    if python_version:
        candidate = shutil.which(f"python{python_version}")
        if candidate:
            return candidate
    return sys.executable


_SYSTEM_TOOLS = {"apt-get", "apt", "sudo", "yum", "dnf", "dpkg", "apk", "brew", "add-apt-repository"}


def _pre_install_commands(install_config):
    commands = []
    for step in (install_config.get("pre_install") or []):
        parts = shlex.split(step)
        if parts and parts[0] not in _SYSTEM_TOOLS:
            commands.append(parts)
    return commands


def _pip_commands(install_config, venv_python):
    pip = [venv_python, "-m", "pip", "install", "--disable-pip-version-check", "-q"]
    install = install_config.get("install") or "pip install -e ."
    install = install.replace("pip install -e", "pip install").replace("pip install", "").strip()
    commands = [pip + shlex.split(install)]
    pip_packages = install_config.get("pip_packages") or []
    for package in pip_packages:
        commands.append(pip + [package])
    if not any("pytest" in package for package in pip_packages):
        commands.append(pip + ["pytest"])
    return commands


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
        return {"python": venv_python, "test_env": _env_from(base, install_config)}
    if not force and fail_marker.exists():
        return None

    base.mkdir(parents=True, exist_ok=True)
    checkout = base / "checkout"
    remote = f"https://github.com/{repo}.git"
    try:
        if not (checkout / ".git").exists():
            _, code = _clone(remote, checkout)
            if code != 0:
                return _fail(base, fail_marker)
            _run(["git", "-C", str(checkout), "checkout", setup_commit], timeout=120)
        interpreter = _interpreter(install_config.get("python"))
        if _run([interpreter, "-m", "venv", str(venv)], timeout=300).returncode != 0:
            return _fail(base, fail_marker)
        _run([venv_python, "-m", "pip", "install", "--disable-pip-version-check", "-q",
              "-U", "pip", "setuptools", "wheel"], timeout=600)
        for command in _pre_install_commands(install_config):
            _run(command, cwd=str(checkout), timeout=600)
        for command in _pip_commands(install_config, venv_python):
            result = _run(command, cwd=str(checkout), timeout=1800)
            if result.returncode != 0:
                (base / "install.log").write_text((result.stdout + result.stderr)[-4000:])
                return _fail(base, fail_marker)
    except Exception as exc:
        (base / "install.log").write_text(str(exc))
        return _fail(base, fail_marker)

    shutil.rmtree(checkout, ignore_errors=True)
    ok_marker.write_text("ok")
    return {"python": venv_python, "test_env": _env_from(base, install_config)}


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


def _fail(base, fail_marker):
    fail_marker.write_text("failed")
    return None


def provision_episode(episode, instance, force=False):
    result = ensure_repo_venv(instance, force=force)
    if result is None:
        return False
    episode["repo_state"]["test_env"] = result["test_env"]
    return True
