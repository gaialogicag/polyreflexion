"""Role to client dispatch.

The registry is the only place that knows which provider serves a role, which
is what makes swapping providers a config change. No test here calls
``complete()``: dispatch is decided by config type, and a real call would need
a network and a key.
"""

from __future__ import annotations

import importlib.util

import pytest

from polyrx.config import (
    BackendsConfig,
    BackendsSchema,
    GeminiConfig,
    ModelConfig,
    OllamaConfig,
    OpenAIConfig,
    PostProcessConfig,
)
from polyrx.models import registry as registry_module
from polyrx.models.ollama import OllamaClient
from polyrx.models.openai_client import OpenAIClient
from polyrx.models.registry import (
    ClientRegistry,
    active_backends,
    get_client,
    set_active_backends,
)


def _gemini_sdk_installed() -> bool:
    """``find_spec`` raises rather than returning None when the parent
    package is absent, which is exactly the case this has to detect."""
    try:
        return importlib.util.find_spec("google.genai") is not None
    except ModuleNotFoundError:
        return False


gemini_sdk_installed = _gemini_sdk_installed()


@pytest.fixture
def openai_key(hermetic_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """A syntactically valid key. Constructing the SDK client opens no
    connection, so a placeholder is enough to reach the dispatch under test.

    Depends on ``hermetic_env`` so the scrub cannot run after this and undo it.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")


class TestDispatchByConfigType:
    def test_openai_config_builds_an_openai_client(self, openai_key: None) -> None:
        backends = BackendsConfig(answerer=OpenAIConfig(model="gpt-4o-mini"))

        client = ClientRegistry(backends).get("answerer")

        assert isinstance(client, OpenAIClient)
        assert client.model == "gpt-4o-mini"

    def test_ollama_config_builds_an_ollama_client(self) -> None:
        backends = BackendsConfig(answerer=OllamaConfig(model="phi4-mini-reasoning"))

        client = ClientRegistry(backends).get("answerer")

        assert isinstance(client, OllamaClient)

    def test_gemini_config_builds_a_gemini_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "test-not-a-real-key")
        backends = BackendsConfig(answerer=GeminiConfig(model="gemini-2.5-flash"))

        if gemini_sdk_installed:
            from polyrx.models.gemini_client import GeminiClient

            assert isinstance(ClientRegistry(backends).get("answerer"), GeminiClient)
        else:
            # Reaching the SDK import proves dispatch chose GeminiClient. The
            # install hint is the documented behaviour on a base install.
            with pytest.raises(ImportError, match="google-genai"):
                ClientRegistry(backends).get("answerer")

    @pytest.mark.parametrize(
        "config",
        [
            pytest.param(OpenAIConfig(), id="openai"),
            pytest.param(GeminiConfig(), id="gemini"),
            pytest.param(OllamaConfig(), id="ollama"),
        ],
    )
    def test_concrete_configs_are_matched_before_the_bare_fallback(
        self, config: ModelConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every provider config subclasses ``ModelConfig``. If the bare
        ``ModelConfig`` branch is ever tested first it swallows all of them,
        and a fully configured run fails claiming no provider was selected."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
        monkeypatch.setenv("GEMINI_API_KEY", "test-not-a-real-key")
        backends = BackendsConfig(answerer=config)

        try:
            ClientRegistry(backends).get("answerer")
        except ImportError:
            # The optional Gemini SDK is absent. Dispatch still resolved.
            pass
        except TypeError as exc:  # pragma: no cover - the regression guarded
            pytest.fail(f"{type(config).__name__} fell through to the fallback: {exc}")

    def test_default_backends_put_the_answerer_on_openai(self, openai_key: None) -> None:
        assert isinstance(ClientRegistry().get("answerer"), OpenAIClient)


class TestRejectedConfigurations:
    def test_unknown_role_lists_the_known_roles(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            ClientRegistry().get("summariser")

        message = str(excinfo.value)
        assert "summariser" in message
        assert "answerer" in message
        assert "judge" in message

    def test_a_role_with_no_provider_selected_is_rejected(self) -> None:
        """``BackendsSchema`` leaves every role a bare ``ModelConfig``. That
        means the config tree named no provider, which must fail loudly rather
        than default to one."""
        with pytest.raises(TypeError) as excinfo:
            ClientRegistry(BackendsSchema()).get("answerer")

        message = str(excinfo.value)
        assert "answerer" in message
        assert "no provider selected" in message

    def test_the_error_names_the_override_that_fixes_it(self) -> None:
        with pytest.raises(TypeError, match=r"backends/role@answerer"):
            ClientRegistry(BackendsConfig(answerer=ModelConfig())).get("answerer")

    def test_a_missing_api_key_is_reported_by_name(
        self, monkeypatch: pytest.MonkeyPatch, isolated_cwd: object
    ) -> None:
        """Run from a directory with no ``.env`` to load, so the only source of
        a key is the environment the fixture already scrubbed."""
        backends = BackendsConfig(answerer=OpenAIConfig(api_key_env="POLYRX_TEST_KEY"))

        with pytest.raises(ValueError, match="POLYRX_TEST_KEY"):
            ClientRegistry(backends).get("answerer")


class TestCaching:
    def test_a_role_is_built_once(self, openai_key: None) -> None:
        """Each construction opens a connection pool, and a benchmark asks for
        the same role once per item."""
        client_registry = ClientRegistry()

        assert client_registry.get("answerer") is client_registry.get("answerer")

    def test_registries_do_not_share_clients(self, openai_key: None) -> None:
        assert ClientRegistry().get("answerer") is not ClientRegistry().get("answerer")

    def test_extraction_settings_reach_the_ollama_client(self) -> None:
        postprocess = PostProcessConfig()
        postprocess.extraction.max_bold_words = 3
        backends = BackendsConfig(answerer=OllamaConfig())

        client = ClientRegistry(backends, postprocess).get("answerer")

        assert client.extraction.max_bold_words == 3


class TestDescribe:
    def test_describe_maps_every_role_to_its_model(self) -> None:
        described = ClientRegistry(
            BackendsConfig(
                answerer=OpenAIConfig(model="gpt-4o-mini"),
                judge=OpenAIConfig(model="gpt-5-mini"),
            )
        ).describe()

        assert described["answerer"] == "gpt-4o-mini"
        assert described["judge"] == "gpt-5-mini"

    def test_describe_does_not_build_clients(self) -> None:
        """It reports what a run *would* call, so it has to work before any
        key is set -- the doctor calls it to print the plan."""
        assert ClientRegistry(BackendsSchema()).describe().keys() >= {"answerer", "judge"}


class TestProcessWideRegistry:
    def test_set_active_backends_installs_the_registry_get_client_reads(
        self, restore_active_registry: None
    ) -> None:
        installed = set_active_backends(BackendsConfig(answerer=OllamaConfig()))

        assert active_backends() is installed
        assert isinstance(get_client("answerer"), OllamaClient)

    def test_installing_a_registry_replaces_the_previous_one(
        self, restore_active_registry: None
    ) -> None:
        first = set_active_backends(BackendsConfig(answerer=OllamaConfig()))
        second = set_active_backends(BackendsConfig(answerer=OllamaConfig()))

        assert active_backends() is second
        assert active_backends() is not first

    def test_the_fixture_restores_the_global(self) -> None:
        """Guards the fixture itself: without it, whichever test ran last
        would decide the backends every later test builds."""
        assert active_backends() is registry_module._active
