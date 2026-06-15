from __future__ import annotations

import logging
import re

from dataclasses import dataclass, is_dataclass
from typing import Dict, Callable, Type
from ulid import ULID

from pypeh.core.models.identifiers import (
    IdentifierContext,
    IdentifierProvider,
)

logger = logging.getLogger(__name__)


class PrefixMap:
    def __init__(self, prefixes: Dict[str, str]):
        """
        example_prefixes = {
            "foaf": "https://xmlns.com/foaf/0.1/",
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
            "owl": "http://www.w3.org/2002/07/owl#",
            "schema": "https://schema.org/",
        }
        """
        self.prefixes = prefixes
        self.reverse_map = {uri: prefix for prefix, uri in prefixes.items()}

    def expand(self, curie: str) -> str:
        """Expand a CURIE to a full URI."""
        prefix, suffix = curie.split(":", 1)
        if prefix in self.prefixes:
            return f"{self.prefixes[prefix]}{suffix}"
        raise ValueError(f"Unknown prefix: {prefix}")

    def compress(self, uri: str) -> str:
        """Compress a full URI to a CURIE if possible."""
        for ns, prefix in self.reverse_map.items():
            if uri.startswith(ns):
                return f"{prefix}:{uri[len(ns):]}"
        return uri  # fallback to full URI if no prefix match


class ImportMapTrieNode:
    def __init__(self):
        self.children = {}
        self.connection_str: str | None = None


class ImportMap:
    """
    Implementation of a trie with dict-like behaviour. For any namespace the
    closest matching namespace and its connection string will be returned.
    """

    def __init__(self):
        self.root = ImportMapTrieNode()
        self._data = set()

    def insert(self, namespace, connection_str):
        self._data.add(namespace)
        parts = self._split_namespace(namespace)
        node = self.root
        for part in parts:
            if part not in node.children:
                node.children[part] = ImportMapTrieNode()
            node = node.children[part]
        node.connection_str = connection_str

    def match(self, uri_or_curie):
        # TODO: if curie first convert to uri
        parts = self._split_namespace(uri_or_curie)
        node = self.root
        last_match = None
        for part in parts:
            if part in node.children:
                node = node.children[part]
                if node.connection_str:
                    last_match = node.connection_str
            else:
                break
        return last_match

    def __getitem__(self, key):
        return self.match(key)

    def __setitem__(self, key, value):
        return self.insert(key, value)

    def __contains__(self, key):
        value = self.match(key)
        return value is not None

    def get(self, key, default=None):
        ret = self.match(key)
        if ret is None:
            return default
        return ret

    def keys(self):
        return list(self._data)

    def values(self):
        return [self.match(key) for key in self._data]

    def items(self):
        return {key: self.match(key) for key in self._data}

    def __iter__(self):
        return iter(self.keys())

    def _split_namespace(self, uri):
        # Split only on "/" and "#"
        return [p for p in uri.replace("#", "/").split("/") if p]


class NamespaceManager:
    def __init__(self, default_base_uri: str | None = None):
        self._default_base_uri = self._validate_and_normalize_base(
            default_base_uri
        )
        self.namespaces: dict[str, str] = {}  # namespace_label -> base_uri
        self.dataclass_namespace_map: dict[
            Type, str
        ] = {}  # dataclass → namespace_label
        self.suffix_strategy: Callable[[], str] = self.generate_ulid()
        self.resource_type_strategy: Callable[[Type], str] = (
            default_resource_type
        )

    @property
    def default_base_uri(self):
        return self._default_base_uri

    def set_suffix_strategy(self, strategy: Callable):
        self.suffix_strategy = strategy

    @staticmethod
    def _validate_and_normalize_base(uri: str | None) -> str | None:
        if uri is None:
            return None
        uri = uri.strip()
        if uri.endswith("/") or uri.endswith("#"):
            return uri
        raise ValueError('default_base_uri should end in "#" or "/"')

    def bind(self, namespace_key: str, base_uri: str):
        self.namespaces[namespace_key] = base_uri.rstrip("/") + "/"

    def register_class(self, cls: Type, namespace: str):
        if not is_dataclass(cls):
            raise TypeError(f"{cls} is not a dataclass")
        if namespace not in self.namespaces:
            raise ValueError(
                f"Namespace {namespace} not bound to NamespaceManager"
            )
        self.dataclass_namespace_map[cls] = namespace

    @classmethod
    def generate_ulid(cls, length: int = 26):
        def _generate_ulid() -> str:
            ret = str(ULID())
            return ret[:length]

        return _generate_ulid

    def _resolve_base(
        self,
        resource_class: Type | None = None,
        namespace_key: str | None = None,
    ) -> str | None:
        # Explicit namespace overrides everything
        if namespace_key is not None:
            base = self.namespaces.get(namespace_key)
            if base is None:
                raise ValueError(
                    f"No registered base URI for namespace {namespace_key}"
                )
            return base

        resource_type = None
        # Class-specific namespace
        if resource_class is not None:
            if resource_class in self.dataclass_namespace_map:
                ns = self.dataclass_namespace_map[resource_class]
                return self.namespaces[ns]
            else:
                resource_type = self.resource_type_strategy(resource_class)

        # Default namespace
        if self.default_base_uri is not None:
            if resource_type is not None:
                return f"{self.default_base_uri}{resource_type}/"
            else:
                return self.default_base_uri

        return None

    def mint(
        self,
        resource_class: Type,
        namespace_key: str | None = None,
        identifying_field: str = "id",
        suffix_strategy: Callable[[], str] | None = None,
    ) -> str:
        base = self._resolve_base(resource_class, namespace_key)
        if base is None:
            raise ValueError("Could not resolve base URI")
        suffix = (
            self.suffix_strategy()
            if suffix_strategy is None
            else suffix_strategy()
        )
        return f"{base}{suffix}"

    def mint_and_set(
        self,
        obj,
        namespace_key: str | None = None,
        identifying_field: str = "id",
    ) -> str:
        uri = self.mint(
            resource_class=obj.__class__,
            namespace_key=namespace_key,
            identifying_field=identifying_field,
        )
        setattr(obj, identifying_field, uri)

        return uri

    def get_identifier_provider(
        self,
        namespace_key: str | None = None,
        suffix_strategy: Callable[[], str] | None = None,
    ) -> IdentifierProvider:
        return NamespaceIdentifierProvider(
            namespace_manager=self,
            namespace_key=namespace_key,
            suffix_strategy=suffix_strategy,
        )


@dataclass
class NamespaceIdentifierProvider:
    namespace_manager: NamespaceManager
    namespace_key: str | None = None
    suffix_strategy: Callable[[], str] | None = None

    def mint(self, context: IdentifierContext) -> str:
        return self.namespace_manager.mint(
            resource_class=context.resource_class,
            namespace_key=self.namespace_key,
            suffix_strategy=self.suffix_strategy,
        )


def default_resource_type(cls: Type) -> str:
    # Convert CamelCase → kebab-case
    name = cls.__name__
    return re.sub(r"(?<!^)(?=[A-Z])", "-", name).lower()
