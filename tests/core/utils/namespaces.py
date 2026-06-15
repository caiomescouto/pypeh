import pytest
import re

from peh_model.peh import ObservableProperty

from pypeh import NamespaceManager
from pypeh.core.models.identifiers import IdentifierContext
from pypeh.core.models.internal_data_layout import DatasetSeries


@pytest.fixture
def nm():
    return NamespaceManager()


@pytest.mark.core
class TestNamespaces:
    def test_bind_namespace(self, nm):
        nm.bind("peh", "https://w3id.org/peh")
        assert "peh" in nm.namespaces
        assert nm.namespaces["peh"] == "https://w3id.org/peh/"

    def test_register_class(self, nm):
        nm.namespaces["obsprop"] = "https://w3id.org/test"
        nm.register_class(ObservableProperty, "obsprop")
        assert nm.dataclass_namespace_map[ObservableProperty] == "obsprop"

    def test_register_non_dataclass_raises(self, nm):
        class NotADataClass:
            pass

        with pytest.raises(TypeError):
            nm.register_class(NotADataClass, "project")

    def test_mint_without_namespace_raises(self, nm):
        with pytest.raises(ValueError):
            nm.register_class(ObservableProperty, "project")
        nm.namespaces["obsprop"] = "https://w3id.org/test"
        nm.register_class(ObservableProperty, "obsprop")
        p = ObservableProperty("question")
        iri = nm.mint(p.__class__)
        assert iri.startswith(nm.namespaces["obsprop"])

    def test_mint_without_class_registration_raises(self, nm):
        nm.bind("project", "https://w3id.org/peh")
        p = ObservableProperty("question")
        with pytest.raises(ValueError):
            nm.mint(p.__class__)

    def test_mint_generates_valid_iri(self, nm):
        nm.bind("project", "https://w3id.org/peh")
        nm.register_class(ObservableProperty, "project")
        p = ObservableProperty("question")
        iri = nm.mint(p.__class__)
        assert iri.startswith("https://w3id.org/peh")
        assert len(iri.split("/")[-1]) > 0  # suffix exists

    def test_custom_suffix_strategy(self, nm):
        nm.bind("project", "https://w3id.org/peh")
        nm.register_class(ObservableProperty, "project")
        nm.set_suffix_strategy(lambda: "fixed-id")
        p = ObservableProperty("question")
        iri = nm.mint(p.__class__)
        assert iri.endswith("/fixed-id")

    def test_hash_suffix_strategy(self, nm):
        nm.bind("project", "https://w3id.org/peh")
        nm.register_class(ObservableProperty, "project")
        nm.suffix_strategy = nm.generate_ulid(length=8)
        p = ObservableProperty("question")
        iri = nm.mint(p.__class__)
        suffix = iri.split("/")[-1]
        assert len(suffix) == 8
        # assert hashing is deterministic
        iri2 = nm.mint(p.__class__)
        assert iri == iri2

    def test_mint_and_set(self, nm):
        nm.bind("project", "https://w3id.org/peh")
        nm.register_class(ObservableProperty, "project")
        nm.suffix_strategy = nm.generate_ulid()
        p = ObservableProperty(id="temp", ui_label="question")
        iri = nm.mint_and_set(p)
        assert p.id == iri

    def test_mint_and_set_resource_type(self):
        nm = NamespaceManager("https://w3id.org/example/id/")
        p = ObservableProperty(id="temp", ui_label="question")

        iri = nm.mint_and_set(p)
        pattern = r"^https://w3id\.org/example/id/observable-property/[0-9A-HJKMNP-TV-Z]{26}$"

        assert re.match(
            pattern, iri
        ), f"IRI did not match expected pattern: {iri}"

    def test_identifier_provider_delegates_to_mint(self):
        nm = NamespaceManager("https://w3id.org/peh/")
        provider = nm.get_identifier_provider(
            suffix_strategy=lambda: "fixed-id"
        )

        iri = provider.mint(IdentifierContext(DatasetSeries))

        assert iri == "https://w3id.org/peh/dataset-series/fixed-id"
