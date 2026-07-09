import logging
import pkgutil

from linkml_runtime import SchemaView

logger = logging.getLogger(__name__)

ENTITYLIST_MAPPING = {
    "MatrixSubClass": "matrix_subclasses",
    "ObservablePropertyMetadataField": "metadata_fields",
    "BioChemEntitySubClass": "biochementity_subclasses",
    "Grouping": "groupings",
    "IndicatorSubClass": "indicator_subclasses",
    "ObservableProperty": "observable_properties",
    "Stakeholder": "stakeholders",
    "Project": "projects",
    "Study": "studies",
    "StudyEntity": "study_entities",
    "PhysicalEntity": "physical_entities",
    "ObservationGroup": "observation_groups",
    "Observation": "observations",
    "DerivedObservation": "derived_observations",
    "ObservationDesign": "observation_designs",
    "ObservationResult": "observation_results",
    "ObservedValue": "observed_values",
    "DataLayout": "layouts",
    "DataImportConfig": "import_configs",
    "DataExportConfig": "export_configs",
    "DataRequest": "data_requests",
}


def get_from_entity_list_map(entity_type: str) -> str | None:
    return ENTITYLIST_MAPPING.get(entity_type, None)


def get_schema_view() -> SchemaView:
    schema_text = pkgutil.get_data("peh_model", "schema/peh.yaml")
    assert schema_text is not None
    schema_text = schema_text.decode()
    ret = SchemaView(schema_text)
    assert isinstance(ret, SchemaView)
    return ret
