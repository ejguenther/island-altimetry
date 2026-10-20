from island_altimetry import inventory as inv, sr_client, status


def test_parse_icesat2_and_atl24_names():
    a = inv.parse_name("atl03", "ATL03_20210910031707_11991201_007_01.h5")
    assert (a["rgt"], a["cycle"], a["region"], a["release"]) == (1199, 12, 1, "007")
    assert a["acq_start_name_utc"] == "2021-09-10T03:17:07Z"
    b = inv.parse_name("atl24", "ATL24_20210910031707_11991201_006_02_002_01.h5")
    assert b["release"] == "002" and b["atl03_release"] == "006"
    assert a["lineage_key"] == b["lineage_key"]


def test_parse_gedi_name_keeps_exact_ids():
    g = inv.parse_name("gedi_l2a", "GEDI02_A_2021262155644_O15692_03_T08289_02_003_02_V002.h5")
    assert g["orbit"] == 15692 and g["release"] == "002"
    assert g["acq_start_name_utc"] == "2021-09-19T15:56:44Z"  # day-of-year 262


def test_dedupe_keeps_highest_revision_with_lineage():
    rows = [inv.parse_name("atl03", n) for n in
            ("ATL03_20210910031707_11991201_007_01.h5", "ATL03_20210910031707_11991201_007_02.h5")]
    sel, dis = inv.dedupe(rows)
    assert [r["revision"] for r in sel] == ["02"]
    assert dis[0]["superseded_by"].endswith("_007_02")


def test_windows_cover_interval_without_gaps():
    w = inv.windows("2018-10-13T00:00:00Z", "2021-01-01T00:00:00Z", 366)
    assert w[0][0] == "2018-10-13T00:00:00Z" and w[-1][1] == "2021-01-01T00:00:00Z"
    assert all(a[1] == b[0] for a, b in zip(w, w[1:]))


def test_discovery_failure_is_not_zero(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("HTTP error <503>")
    monkeypatch.setattr(sr_client, "search", boom)
    found, ledger = inv.sliderule_discover("icesat2", [], "2021-01-01T00:00:00Z", "2021-02-01T00:00:00Z",
                                           300, retries=2, backoff_s=0)
    assert found == {} and ledger[-1]["status"] == "retrieval_failed" and ledger[-1]["n"] is None


def test_empty_list_is_no_granules(monkeypatch):
    monkeypatch.setattr(sr_client, "search", lambda *a, **k: [])
    _, ledger = inv.sliderule_discover("icesat2", [], "2021-01-01T00:00:00Z", "2021-02-01T00:00:00Z", 300, 1, 0)
    assert ledger[-1]["status"] == "no_granules" and ledger[-1]["n"] == 0


def test_cap_exceeded_splits_window(monkeypatch):
    calls = []

    def fake(asset, poly, a, b, **k):
        calls.append((a, b))
        if len(calls) == 1:
            raise RuntimeError("number of CMR hits <400> exceeded maximum allowed <300>")
        return {f"G{len(calls)}": {}}
    monkeypatch.setattr(sr_client, "search", fake)
    found, ledger = inv.sliderule_discover("icesat2", [], "2021-01-01T00:00:00Z", "2021-03-01T00:00:00Z", 300, 1, 0)
    assert len(calls) == 3 and len(found) == 2
    assert ledger[0]["status"] == "split_cap_exceeded"


def test_classify_outcomes():
    alerts_fail = [{"code": -7, "level": 4, "text": "Failure on resource X beam gt1l: H5Coro::Future read failure on gt1l/heights/foo"}]
    assert sr_client.classify(alerts_fail, None, None, {"foo"}) == status.FIELD_UNSUPPORTED
    assert sr_client.classify(alerts_fail, None, None, set()) == status.RETRIEVAL_FAILED
    nores = [{"code": -1, "level": 4, "text": "(dataframe.lua:179) <x> has no resources to process"}]
    assert sr_client.classify(nores, None, None) == status.NO_GRANULES
    assert sr_client.classify([], "ConnectionError()", None) == status.RETRIEVAL_FAILED
    assert sr_client.classify([{"code": -1, "level": 4, "text": "Failed to load script x.lua"}], None, None) == status.SERVICE_UNSUPPORTED


def test_missing_atl09_partner_is_no_granules():
    alerts = [{"code": -1, "level": 4, "text": "(icesat2_utils.lua:44) Invalid number of resources for ATL09 CMR request for {...}: 0"},
              {"code": -1, "level": 4, "text": "(dataframe.lua:248) <x> failed to write dataframe"}]
    assert sr_client.classify(alerts, None, None) == status.NO_GRANULES
