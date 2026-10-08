import importlib
import pytest

from dataclasses import fields
from pypeh.core.models.peh_wrappers import ENTITYLIST_MAPPING
from peh_model.peh import EntityList


@pytest.mark.core
class TestWrapper:
    def test_entity_list_map(self):
        entitylist_fields = {f.name for f in fields(EntityList)}

        for cls_name, entity_list_field in ENTITYLIST_MAPPING.items():
            # assert import cls from peh_model.peh
            module = importlib.import_module("peh_model.peh")
            assert hasattr(module, cls_name), (
                f"Class {cls_name} not found in peh_model.peh"
            )

            # assert EntityList has field entity_list_field
            assert entity_list_field in entitylist_fields, (
                f"EntityList is missing field '{entity_list_field}' "
                f"for mapped class '{cls_name}'"
            )
