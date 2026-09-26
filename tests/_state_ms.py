"""Small populated MSv2 for the pinned xarray-ms reader and native writer."""

from __future__ import annotations


def make_state_ms(path, *, set_category=True):
    import numpy as np
    from casacore.tables import default_ms, makearrcoldesc, maketabdesc, table

    with default_ms(str(path), maketabdesc([makearrcoldesc("DATA", 0j, shape=[4, 4], valuetype="complex")])) as main:
        main.addrows(6)
        for name in ("ARRAY_ID", "OBSERVATION_ID", "PROCESSOR_ID", "FEED1", "FEED2", "DATA_DESC_ID", "FIELD_ID", "STATE_ID", "ANTENNA1"):
            main.putcol(name, np.zeros(6, np.int32))
        main.putcol("ANTENNA2", np.ones(6, np.int32))
        main.putcol("SCAN_NUMBER", np.ones(6, np.int32))
        for name in ("TIME", "TIME_CENTROID"):
            main.putcol(name, 60310 * 86400.0 + np.arange(6) * 8)
        for name in ("INTERVAL", "EXPOSURE"):
            main.putcol(name, np.full(6, 8.0))
        main.putcol("UVW", np.arange(18, dtype=float).reshape(6, 3))
        main.putcol("DATA", (np.arange(96).reshape(6, 4, 4) + 0.5j).astype(np.complex64))
        main.putcol("FLAG", np.zeros((6, 4, 4), bool))
        main.putcol("FLAG_ROW", np.array([True, False, False, False, False, False]))
        main.putcol("WEIGHT", np.broadcast_to(np.array([2, 3, 5, 7], np.float32), (6, 4)))
        main.putcol("SIGMA", np.broadcast_to(np.array([1, 2, 3, 4], np.float32), (6, 4)))
        if set_category:
            main.putcolkeyword("FLAG_CATEGORY", "CATEGORY", np.asarray(["test"]))
        main.putcolkeyword("DATA", "QuantumUnits", np.asarray(["Jy"]))
        main.putkeyword("SHINOBI_FIXTURE_PROFILE", "six-row-fixed")

    def fill(name, rows):
        with table(str(path / name), readonly=False, ack=False) as tab:
            tab.addrows(len(rows))
            for index, row in enumerate(rows):
                for key, value in row.items():
                    tab.putcell(key, index, value)

    fill(
        "ANTENNA",
        [
            {
                "NAME": f"m{i:03d}",
                "STATION": f"m{i:03d}",
                "POSITION": np.array([5109360.0 + i * 137, 2006852.0, -3238948.0]),
                "DISH_DIAMETER": 13.5,
                "MOUNT": "ALT-AZ",
                "TYPE": "GROUND-BASED",
                "FLAG_ROW": False,
            }
            for i in range(2)
        ],
    )
    fill(
        "FEED",
        [
            {
                "ANTENNA_ID": i,
                "FEED_ID": 0,
                "SPECTRAL_WINDOW_ID": 0,
                "TIME": 60310 * 86400.0,
                "INTERVAL": 1e30,
                "NUM_RECEPTORS": 2,
                "BEAM_ID": -1,
                "BEAM_OFFSET": np.zeros((2, 2)),
                "POLARIZATION_TYPE": ["X", "Y"],
                "POL_RESPONSE": np.eye(2, dtype=np.complex64),
                "POSITION": np.zeros(3),
                "RECEPTOR_ANGLE": np.zeros(2),
            }
            for i in range(2)
        ],
    )
    fill(
        "SPECTRAL_WINDOW",
        [
            {
                "NUM_CHAN": 4,
                "CHAN_FREQ": 1.4e9 + np.arange(4) * 1e7,
                "CHAN_WIDTH": np.full(4, 1e7),
                "EFFECTIVE_BW": np.full(4, 1e7),
                "RESOLUTION": np.full(4, 1e7),
                "REF_FREQUENCY": 1.4e9,
                "TOTAL_BANDWIDTH": 4e7,
                "NAME": "SPW0",
                "MEAS_FREQ_REF": 5,
                "FREQ_GROUP_NAME": "Group1",
            }
        ],
    )
    fill("POLARIZATION", [{"NUM_CORR": 4, "CORR_TYPE": np.array([9, 10, 11, 12]), "CORR_PRODUCT": np.array([[0, 0, 1, 1], [0, 1, 0, 1]])}])
    fill("DATA_DESCRIPTION", [{"SPECTRAL_WINDOW_ID": 0, "POLARIZATION_ID": 0, "FLAG_ROW": False}])
    direction = np.array([[0.35, -0.52]])
    fill(
        "FIELD",
        [{"NAME": "fixture", "CODE": "", "PHASE_DIR": direction, "DELAY_DIR": direction, "REFERENCE_DIR": direction, "SOURCE_ID": 0, "TIME": 60310 * 86400.0, "NUM_POLY": 0}],
    )
    fill("STATE", [{"OBS_MODE": "OBSERVE_TARGET#ON_SOURCE", "SIG": True, "REF": False, "FLAG_ROW": False}])
    fill("OBSERVATION", [{"TELESCOPE_NAME": "MEERKAT", "OBSERVER": "shinobi-tests", "PROJECT": "state-tests", "TIME_RANGE": np.array([60310 * 86400.0, 60310 * 86400.0 + 48])}])
    fill("PROCESSOR", [{"TYPE": "CORRELATOR", "SUB_TYPE": "test", "MODE_ID": 0, "FLAG_ROW": False}])
    return path
