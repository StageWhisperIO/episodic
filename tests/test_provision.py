import os

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
