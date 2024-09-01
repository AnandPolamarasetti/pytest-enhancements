import json
import os
import posixpath
import re
import shutil
import subprocess
import sys
import threading
import traceback

import click
import pytest

import mlflow
from mlflow.environment_variables import _MLFLOW_TESTING, MLFLOW_TRACKING_URI
from mlflow.utils.os import is_windows
from mlflow.version import VERSION

from tests.helper_functions import get_safe_port


def pytest_addoption(parser):
    parser.addoption(
        "--requires-ssh",
        action="store_true",
        dest="requires_ssh",
        default=False,
        help="Run tests decorated with 'requires_ssh' annotation. "
             "These tests require keys to be configured locally "
             "for SSH authentication.",
    )
    parser.addoption(
        "--ignore-flavors",
        action="store_true",
        dest="ignore_flavors",
        default=False,
        help="Ignore tests for model flavors.",
    )
    parser.addoption(
        "--splits",
        default=None,
        type=int,
        help="The number of groups to split tests into.",
    )
    parser.addoption(
        "--group",
        default=None,
        type=int,
        help="The group of tests to run.",
    )
    parser.addoption(
        "--serve-wheel",
        action="store_true",
        default=os.getenv("CI", "false").lower() == "true",
        help="Serve a wheel for the dev version of MLflow. True by default in CI, False otherwise.",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_ssh")
    config.addinivalue_line("markers", "notrackingurimock")
    config.addinivalue_line("markers", "allow_infer_pip_requirements_fallback")
    config.addinivalue_line("markers", "do_not_disable_new_import_hook_firing_if_module_already_exists")
    config.addinivalue_line("markers", "classification")

    labels = fetch_pr_labels() or []
    if "fail-fast" in labels:
        config.option.maxfail = 1


@pytest.hookimpl(tryfirst=True)
def pytest_cmdline_main(config):
    group = config.getoption("group")
    splits = config.getoption("splits")

    if splits is None and group is None:
        return None

    if splits and group is None:
        raise pytest.UsageError("`--group` is required")

    if group and splits is None:
        raise pytest.UsageError("`--splits` is required")

    if splits < 1:
        raise pytest.UsageError("`--splits` must be >= 1")

    if group < 1 or group > splits:
        raise pytest.UsageError(f"`--group` must be between 1 and {splits}")

    return None


def pytest_sessionstart(session):
    if uri := MLFLOW_TRACKING_URI.get():
        click.echo(
            click.style(
                f"Environment variable {MLFLOW_TRACKING_URI} is set to {uri!r}, which may interfere with tests.",
                fg="red",
            )
        )


def pytest_runtest_setup(item):
    markers = [mark.name for mark in item.iter_markers()]
    if "requires_ssh" in markers and not item.config.getoption("--requires-ssh"):
        pytest.skip("use `--requires-ssh` to run this test")


def fetch_pr_labels():
    """
    Returns the labels associated with the current pull request.
    """
    if "GITHUB_ACTIONS" not in os.environ:
        return None

    if os.environ.get("GITHUB_EVENT_NAME") != "pull_request":
        return None

    try:
        with open(os.environ["GITHUB_EVENT_PATH"]) as f:
            pr_data = json.load(f)
            return [label["name"] for label in pr_data["pull_request"]["labels"]]
    except (json.JSONDecodeError, FileNotFoundError):
        return []


@pytest.hookimpl(hookwrapper=True)
def pytest_report_teststatus(report, config):
    outcome = yield
    if report.when == "call":
        try:
            import psutil # type: ignore
        except ImportError:
            return

        (*rest, result) = outcome.get_result()
        mem = psutil.virtual_memory()
        mem_used = mem.used / 1024**3
        mem_total = mem.total / 1024**3

        disk = psutil.disk_usage("/")
        disk_used = disk.used / 1024**3
        disk_total = disk.total / 1024**3
        outcome.force_result(
            (
                *rest,
                (
                    f"{result} | "
                    f"MEM {mem_used:.1f}/{mem_total:.1f} GB | "
                    f"DISK {disk_used:.1f}/{disk_total:.1f} GB"
                ),
            )
        )


@pytest.hookimpl(hookwrapper=True)
def pytest_ignore_collect(collection_path, config):
    outcome = yield
    if not outcome.get_result() and config.getoption("ignore_flavors"):
        # If not ignored by the default hook and `--ignore-flavors` specified
        model_flavors = set([
            "tests/autogen",
            "tests/azureml",
            "tests/catboost",
            "tests/diviner",
            "tests/fastai",
            "tests/gluon",
            "tests/h2o",
            "tests/johnsnowlabs",
            "tests/keras",
            "tests/keras_core",
            "tests/llama_index",
            "tests/langchain",
            "tests/lightgbm",
            "tests/mleap",
            "tests/models",
            "tests/onnx",
            "tests/openai",
            "tests/paddle",
            "tests/pmdarima",
            "tests/promptflow",
            "tests/prophet",
            "tests/pyfunc",
            "tests/pytorch",
            "tests/sagemaker",
            "tests/sentence_transformers",
            "tests/shap",
            "tests/sklearn",
            "tests/spacy",
            "tests/spark",
            "tests/statsmodels",
            "tests/tensorflow",
            "tests/transformers",
            "tests/xgboost",
            "tests/test_mlflow_lazily_imports_ml_packages.py",
            "tests/utils/test_model_utils.py",
            "tests/tracking/fluent/test_fluent_autolog.py",
            "tests/autologging/test_autologging_safety_unit.py",
            "tests/autologging/test_autologging_behaviors_unit.py",
            "tests/autologging/test_autologging_behaviors_integration.py",
            "tests/autologging/test_autologging_utils.py",
            "tests/autologging/test_training_session.py",
            "tests/server/auth",
            "tests/gateway",
        ])

        relpath = os.path.relpath(str(collection_path)).replace(os.sep, posixpath.sep)  # for Windows

        if relpath in model_flavors:
            outcome.force_result(True)


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(session, config, items):
    # Executing `tests.server.test_prometheus_exporter` after `tests.server.test_handlers`
    # results in an error because Flask >= 2.2.0 doesn't allow calling setup method such as
    # `before_request` on the application after the first request. To avoid this issue,
    # execute `tests.server.test_prometheus_exporter` first by reordering the test items.
    items.sort(key=lambda item: item.module.__name__ != "tests.server.test_prometheus_exporter")

    # Select the tests to run based on the group and splits
    if (splits := config.getoption("--splits")) and (group := config.getoption("--group")):
        items[:] = items[(group - 1)::splits]


@pytest.hookimpl(hookwrapper=True)
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    yield
    failed_test_reports = terminalreporter.stats.get("failed", [])
    if failed_test_reports:
        if len(failed_test_reports) <= 30:
            terminalreporter.section("Command to run failed test cases")
            ids = [repr(report.nodeid) for report in failed_test_reports]
        else:
            terminalreporter.section("Command to run failed test suites")
            ids = list(dict.fromkeys(report.fspath for report in failed_test_reports))
        terminalreporter.write(" ".join(["pytest"] + ids))
        terminalreporter.write("\n")


@pytest.fixture(scope="session")
def serve_wheel(request):
    """
    Fixture to serve the wheel for the dev version of MLflow.

    This fixture runs in the background and serves the wheel from the specified
    path to be used by the tests.
    """
    if not request.config.getoption("serve_wheel"):
        yield None
        return

    wheel_path = posixpath.join(
        os.path.dirname(os.path.abspath(__file__)), "dist", "mlflow-{}.tar.gz".format(VERSION)
    )

    if not os.path.exists(wheel_path):
        raise RuntimeError(f"Wheel file {wheel_path} does not exist")

    def _serve():
        try:
            subprocess.check_output(["python", "-m", "http.server", "--bind", "localhost", "--directory", os.path.dirname(wheel_path)], stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            print(f"Failed to serve wheel: {e.output.decode()}", file=sys.stderr)
        except Exception as e:
            print(f"Unexpected error serving wheel: {e}", file=sys.stderr)

    thread = threading.Thread(target=_serve, daemon=True)
    thread.start()

    yield wheel_path

    try:
        subprocess.run(["pkill", "-f", "python -m http.server"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to stop server: {e}", file=sys.stderr)
    except Exception as e:
        print(f"Unexpected error stopping server: {e}", file=sys.stderr)

    thread.join()


@pytest.fixture(scope="session", autouse=True)
def clean_up_envs():
    """
    Fixture to clean up environment variables after the test session.
    """
    try:
        virtualenv_path = mlflow.utils.virtualenv._get_mlflow_virtualenv_root()
        if virtualenv_path and os.path.exists(virtualenv_path):
            shutil.rmtree(virtualenv_path)
    except Exception:
        print("Error while cleaning up virtual environments:", file=sys.stderr)
        traceback.print_exc()
