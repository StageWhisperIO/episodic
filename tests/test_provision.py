import os
import sys

from episodic import replay
from episodic.eval import provision


def test_pip_commands_strip_editable_and_ensure_pytest():
    config = {"install": "pip install -e .[flask,starlette]", "pip_packages": ["numpy"]}
    commands = provision._pip_commands(config, "/venv/bin/python")
    assert commands[0][-1] == ".[flask,starlette]"
    assert all("-e" not in command for command in commands)
    assert commands[-1][-1] == "pytest"


def test_pip_commands_do_not_duplicate_pytest():
    config = {"install": "pip install -e .", "pip_packages": ["pytest-cov", "pytest"]}
    commands = provision._pip_commands(config, "/venv/bin/python")
    assert sum(command[-1] == "pytest" for command in commands) == 1
    assert any(command[-1] == "pytest-cov" for command in commands)


def test_pre_install_skips_system_package_managers():
    config = {"pre_install": ["apt-get install -y libxml2", "sudo yum install foo",
                              "python setup.py build_ext --inplace"]}
    commands = provision._pre_install_commands(config)
    assert commands == [["python", "setup.py", "build_ext", "--inplace"]]


def test_slug_is_stable_and_repo_scoped():
    a = provision._slug("acme/widgets", "1.0")
    assert a == provision._slug("acme/widgets", "1.0")
    assert a != provision._slug("acme/widgets", "2.0")
    assert a.startswith("acme__widgets_")


def test_test_environment_prepends_workspace_to_pythonpath(tmp_path):
    (tmp_path / "src").mkdir()
    env = replay._test_environment(tmp_path, {"FOO": "bar", "PATH": "/custom/bin"})
    assert env["FOO"] == "bar"
    assert env["PATH"] == "/custom/bin"
    roots = env["PYTHONPATH"].split(os.pathsep)
    assert roots[0] == str(tmp_path)
    assert str(tmp_path / "src") in roots


def test_test_environment_without_extra_env(tmp_path):
    env = replay._test_environment(tmp_path, None)
    assert env["PYTHONPATH"].split(os.pathsep)[0] == str(tmp_path)


def test_pip_commands_no_deps_prepends_flag_without_dropping_extras():
    config = {"install": "pip install -e .[dev]"}
    commands = provision._pip_commands(config, "/venv/bin/python", no_deps=True)
    assert commands[0][-2:] == ["--no-deps", ".[dev]"]


def test_pip_commands_default_has_no_no_deps_flag():
    config = {"install": "pip install -e ."}
    commands = provision._pip_commands(config, "/venv/bin/python")
    assert "--no-deps" not in commands[0]


def test_clean_frozen_requirements_drops_self_editable_and_local_paths():
    text = "\n".join([
        "requests==2.31.0",
        "-e git+https://github.com/acme/widgets.git@abc123#egg=widgets",
        "iniconfig @ file:///croot/iniconfig_1610983019677/work",
        "# a comment",
        "",
        "numpy==1.26.4",
    ])
    cleaned = provision._clean_frozen_requirements(text)
    assert cleaned == "requests==2.31.0\nnumpy==1.26.4"


def test_clean_frozen_requirements_empty_when_nothing_installable():
    text = "-e git+https://github.com/acme/widgets.git@abc123#egg=widgets\n"
    assert provision._clean_frozen_requirements(text) == ""


def test_pyenv_versions_lists_dirs_sorted_descending(tmp_path, monkeypatch):
    monkeypatch.setenv("PYENV_ROOT", str(tmp_path))
    for name in ("3.9.6", "3.12.3", "3.9.18"):
        (tmp_path / "versions" / name / "bin").mkdir(parents=True)
    assert provision._pyenv_versions() == ["3.12.3", "3.9.18", "3.9.6"]


def test_matching_pyenv_python_prefers_prefix_match(tmp_path, monkeypatch):
    monkeypatch.setenv("PYENV_ROOT", str(tmp_path))
    bin_dir = tmp_path / "versions" / "3.9.18" / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "python").write_text("")
    assert provision._matching_pyenv_python("3.9") == str(bin_dir / "python")
    assert provision._matching_pyenv_python("3.11") is None


def test_resolve_interpreter_uses_pyenv_match_without_mismatch(tmp_path, monkeypatch):
    monkeypatch.setenv("PYENV_ROOT", str(tmp_path))
    bin_dir = tmp_path / "versions" / "3.9.18" / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "python").write_text("")
    interpreter, mismatch = provision._resolve_interpreter("3.9")
    assert interpreter == str(bin_dir / "python")
    assert mismatch is None


def test_resolve_interpreter_records_mismatch_when_version_unavailable(tmp_path, monkeypatch):
    monkeypatch.setenv("PYENV_ROOT", str(tmp_path))
    monkeypatch.delenv("PATH", raising=False)
    monkeypatch.setenv("PATH", "/nonexistent-bin-only")
    interpreter, mismatch = provision._resolve_interpreter("3.9")
    assert interpreter == sys.executable
    assert mismatch is not None
    assert "3.9" in mismatch


def test_resolve_interpreter_no_requested_version_is_not_a_mismatch():
    interpreter, mismatch = provision._resolve_interpreter(None)
    assert interpreter == sys.executable
    assert mismatch is None
