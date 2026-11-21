from importlib.metadata import metadata, version

import sigline as sl


def test_version_matches_installed_metadata():
    assert sl.__version__ == version("sigline")


def test_metadata_name_and_license():
    meta = metadata("sigline")
    assert meta["Name"] == "sigline"
    assert meta["License-Expression"] == "MIT"
