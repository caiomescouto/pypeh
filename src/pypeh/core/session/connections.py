from __future__ import annotations

import logging

from contextlib import AbstractContextManager, contextmanager
from typing import TYPE_CHECKING, Generator

import pypeh.core.models.settings as settingsmodels

from pypeh.adapters.persistence.hosts import (
    HostAdapter,
    S3StorageProvider,
    LocalStorageProvider,
    WebIO,
)

if TYPE_CHECKING:
    import pydantic_settings

logger = logging.getLogger(__name__)


class ConnectionManager:
    def __init__(self, config: settingsmodels.ValidatedImportConfig):
        self._config = config
        self._import_map = config.import_map

    def _register_connection_label(
        self, connection_label: str, settings: pydantic_settings.BaseSettings
    ) -> bool:
        return self._config.register_connection_label(
            connection_label, settings
        )

    @classmethod
    def _create_adapter(
        cls, settings: pydantic_settings.BaseSettings | None, **kwargs
    ) -> HostAdapter:
        if settings is None:
            return WebIO(**kwargs)

        if isinstance(settings, settingsmodels.S3Settings):
            return S3StorageProvider(settings, **kwargs)
        elif isinstance(settings, settingsmodels.LocalFileSettings):
            return LocalStorageProvider(settings, **kwargs)
        else:
            logger.warning(
                f"No adapter registered for settings type: {type(settings)}. Falling back to WebIO."
            )
            return WebIO(**kwargs)

    def get_connection(
        self,
        namespace: str | None = None,
        connection_label: str | None = None,
        **kwargs,
    ) -> AbstractContextManager[HostAdapter]:
        return self._connection_context(
            namespace=namespace,
            connection_label=connection_label,
            **kwargs,
        )

    @contextmanager
    def _connection_context(
        self,
        namespace: str | None = None,
        connection_label: str | None = None,
        **kwargs,
    ) -> Generator[HostAdapter, None, None]:
        settings = self._config.get_settings(
            namespace=namespace, connection_label=connection_label
        )
        if settings is None:
            raise ValueError(
                f"No settings found for connection '{connection_label or namespace}'"
            )

        adapter = self._create_adapter(settings, **kwargs).connect()
        try:
            yield adapter
        finally:
            adapter.close()
