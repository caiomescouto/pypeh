import importlib

import pytest
import peh_model.peh as peh

from pypeh import LocalFileConfig, Session
from pypeh.adapters.persistence.serializations import ExcelIO
from pypeh.core.models.constants import ObservablePropertyValueType
from pypeh.core.models.internal_data_layout import Dataset, DatasetSeries


@pytest.fixture
def export_session(tmp_path):
    return Session(
        connection_config=[
            LocalFileConfig(
                label="local_file",
                config_dict={"root_folder": str(tmp_path)},
            )
        ],
        default_connection=None,
    )


@pytest.fixture
def source_dataset_series():
    pl = importlib.import_module("polars")

    series = DatasetSeries(label="session_series")
    sample = series.add_empty_dataset("SAMPLE")
    sample.add_observation_to_index("peh:obs_sample")
    series.add_observable_property(
        observation_id="peh:obs_sample",
        observable_property_id="peh:prop_id_sample",
        data_type=ObservablePropertyValueType.STRING,
        dataset_label="SAMPLE",
        element_label="id_sample",
        is_primary_key=True,
    )
    sample.data = pl.DataFrame({"id_sample": ["sample-a", "sample-b"]})

    lab = series.add_empty_dataset("LAB")
    lab.add_observation_to_index("peh:obs_lab")
    series.add_observable_property(
        observation_id="peh:obs_lab",
        observable_property_id="peh:prop_id_sample",
        data_type=ObservablePropertyValueType.STRING,
        dataset_label="LAB",
        element_label="id_sample",
    )
    series.add_observable_property(
        observation_id="peh:obs_lab",
        observable_property_id="peh:prop_chol",
        data_type=ObservablePropertyValueType.FLOAT,
        dataset_label="LAB",
        element_label="chol",
    )
    lab.schema.add_foreign_key_link(
        element_label="id_sample",
        foreign_key_dataset_label="SAMPLE",
        foreign_key_element_label="id_sample",
    )
    lab.data = pl.DataFrame(
        {"id_sample": ["sample-a", "sample-b"], "chol": [1.2, 3.4]}
    )

    return series


def _populate_export_cache(session: Session) -> peh.DataExportConfig:
    session.cache.add(
        peh.ObservableProperty(id="peh:prop_id_sample", value_type="string")
    )
    session.cache.add(
        peh.ObservableProperty(id="peh:prop_chol", value_type="float")
    )
    session.cache.add(
        peh.ObservationDesign(
            id="peh:obs_lab_design",
            observable_property_specifications=[
                peh.ObservablePropertySpecification(
                    observable_property="peh:prop_id_sample",
                    specification_category=peh.ObservablePropertySpecificationCategory.identifying,
                ),
                peh.ObservablePropertySpecification(
                    observable_property="peh:prop_chol",
                    specification_category=peh.ObservablePropertySpecificationCategory.required,
                ),
            ],
        )
    )
    session.cache.add(
        peh.Observation(
            id="peh:obs_lab",
            observation_design="peh:obs_lab_design",
        )
    )

    export_section = peh.DataLayoutSection(
        id="peh:LAB_EXPORT_SECTION",
        ui_label="LAB_EXPORT",
        elements=[
            peh.DataLayoutElement(
                label="sample_id",
                observable_property="peh:prop_id_sample",
            ),
            peh.DataLayoutElement(
                label="cholesterol_mg_dl",
                observable_property="peh:prop_chol",
            ),
        ],
    )
    session.cache.add(export_section)
    session.cache.add(
        peh.DataLayout(
            id="peh:LAB_EXPORT_LAYOUT",
            ui_label="LAB_EXPORT_LAYOUT",
            sections=[export_section],
        )
    )

    data_export_config = peh.DataExportConfig(
        id="peh:LAB_EXPORT_CONFIG",
        layout="peh:LAB_EXPORT_LAYOUT",
        section_mapping=peh.DataImportSectionMapping(
            section_mapping_links=[
                peh.DataImportSectionMappingLink(
                    section="peh:LAB_EXPORT_SECTION",
                    observation_id_list=["peh:obs_lab"],
                )
            ]
        ),
    )
    session.cache.add(data_export_config)
    return data_export_config


@pytest.mark.dataframe
class TestSessionExport:
    def test_export_tabular_dataset_series_returns_reshaped_series(
        self, export_session, source_dataset_series
    ):
        data_export_config = _populate_export_cache(export_session)

        exported = export_session.export_tabular_dataset_series(
            source_dataset_series=source_dataset_series,
            data_export_config=data_export_config,
        )

        assert isinstance(exported, DatasetSeries)
        assert set(exported.parts) == {"LAB_EXPORT"}
        export_dataset = exported.parts["LAB_EXPORT"]
        assert isinstance(export_dataset, Dataset)
        assert export_dataset.data is not None
        assert export_dataset.data.shape == (2, 2)
        assert set(export_dataset.data.columns) == {
            "sample_id",
            "cholesterol_mg_dl",
        }
        assert export_dataset.data.get_column("sample_id").to_list() == [
            "sample-a",
            "sample-b",
        ]
        assert export_dataset.data.get_column(
            "cholesterol_mg_dl"
        ).to_list() == [1.2, 3.4]
        assert exported.context_lookup("peh:obs_lab", "peh:prop_chol") == (
            "LAB_EXPORT",
            "cholesterol_mg_dl",
        )

    def test_export_then_dump_parquet_roundtrip(
        self, export_session, source_dataset_series
    ):
        data_export_config = _populate_export_cache(export_session)

        exported = export_session.export_tabular_dataset_series(
            source_dataset_series=source_dataset_series,
            data_export_config=data_export_config,
        )
        source_paths = export_session.dump_tabular_dataset_series(
            dataset_series=exported,
            output_path="export",
            file_format="parquet",
            connection_label="local_file",
        )

        assert len(source_paths) == 1

        loaded = export_session.read_tabular_dataset_series(
            source_paths,
            file_format="parquet",
            connection_label="local_file",
        )

        assert set(loaded.parts) == {"LAB_EXPORT"}
        lab_export = loaded["LAB_EXPORT"]
        assert isinstance(lab_export, Dataset)
        assert lab_export.data is not None
        assert lab_export.data.shape == (2, 2)
        assert set(lab_export.data.columns) == {
            "sample_id",
            "cholesterol_mg_dl",
        }
        assert loaded.context_lookup("peh:obs_lab", "peh:prop_chol") == (
            "LAB_EXPORT",
            "cholesterol_mg_dl",
        )

    def test_reading_two_exported_dataset_series_together_raises(
        self, export_session, source_dataset_series
    ):
        data_export_config = _populate_export_cache(export_session)

        first_export = export_session.export_tabular_dataset_series(
            source_dataset_series=source_dataset_series,
            data_export_config=data_export_config,
        )
        second_export = export_session.export_tabular_dataset_series(
            source_dataset_series=source_dataset_series,
            data_export_config=data_export_config,
        )

        assert first_export.label == second_export.label
        assert first_export.identifier != second_export.identifier

        first_paths = export_session.dump_tabular_dataset_series(
            dataset_series=first_export,
            output_path="first_export",
            file_format="parquet",
            connection_label="local_file",
        )
        second_paths = export_session.dump_tabular_dataset_series(
            dataset_series=second_export,
            output_path="second_export",
            file_format="parquet",
            connection_label="local_file",
        )

        with pytest.raises(
            ValueError,
            match="Parquet files do not belong to the same DatasetSeries",
        ):
            export_session.read_tabular_dataset_series(
                [*first_paths, *second_paths],
                file_format="parquet",
                connection_label="local_file",
            )


@pytest.mark.xlsx
class TestSessionExportXlsx:
    def test_export_then_dump_xlsx(
        self, export_session, source_dataset_series
    ):
        importlib.import_module("xlsxwriter")

        data_export_config = _populate_export_cache(export_session)

        exported = export_session.export_tabular_dataset_series(
            source_dataset_series=source_dataset_series,
            data_export_config=data_export_config,
        )
        source_paths = export_session.dump_tabular_dataset_series(
            dataset_series=exported,
            output_path="export.xlsx",
            file_format="xlsx",
            connection_label="local_file",
        )

        assert len(source_paths) == 1
        workbook = ExcelIO().load(source_paths[0])
        assert set(workbook) == {"LAB_EXPORT"}
        export_sheet = workbook["LAB_EXPORT"]
        assert export_sheet.shape == (2, 2)
        assert set(export_sheet.columns) == {
            "sample_id",
            "cholesterol_mg_dl",
        }
        assert export_sheet.get_column("sample_id").to_list() == [
            "sample-a",
            "sample-b",
        ]
        assert export_sheet.get_column("cholesterol_mg_dl").to_list() == [
            1.2,
            3.4,
        ]
