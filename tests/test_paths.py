import importlib


def test_data_dir_can_be_overridden(monkeypatch, tmp_path):
    monkeypatch.setenv("WORLD_MAPS_DATA", str(tmp_path))
    import world_maps.paths as paths

    paths = importlib.reload(paths)
    assert paths.DATA == tmp_path
    assert paths.EVENT_VIDEOS == tmp_path / "events" / "Exp1_TemporalProbe_Videos"
    assert paths.WOVEN == tmp_path / "woven"
