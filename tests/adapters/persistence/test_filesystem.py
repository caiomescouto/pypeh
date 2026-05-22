import fsspec

from pypeh.adapters.persistence.filesystem import (
    ensure_filesystem_directory,
    ensure_filesystem_parent_directory,
    filesystem_parent,
    join_filesystem_path,
)


def test_join_filesystem_path_preserves_absolute_prefix():
    file_system = fsspec.filesystem("file")

    assert join_filesystem_path(file_system, "/tmp", "pypeh", "file.xlsx") == (
        "/tmp/pypeh/file.xlsx"
    )


def test_filesystem_parent():
    file_system = fsspec.filesystem("memory")

    assert filesystem_parent(file_system, "bucket/path/file.xlsx") == (
        "bucket/path"
    )
    assert filesystem_parent(file_system, "file.xlsx") == ""


def test_ensure_filesystem_directories():
    file_system = fsspec.filesystem("memory")

    ensure_filesystem_directory(file_system, "bucket/data")
    ensure_filesystem_parent_directory(file_system, "bucket/export/file.xlsx")

    assert file_system.isdir("bucket/data")
    assert file_system.isdir("bucket/export")
