"""
The streaming filter's stored triggers: a ranking that recomputes exactly
from what is stored, and a strict file format (a schema version, the ranking's
definition and the bank's identity, each checked on load; no NaN or infinity
stored).
"""
import json
import zipfile

import numpy as np
import pytest

from gravyflow.src.detection import TemplateBank
from gravyflow.src.detection.background import fit_templates
from gravyflow.src.detection.stream import (
    TRIGGER_SCHEMA_VERSION, DetectorTriggers, bank_identity, coincide)
from gravyflow.src.detection.vetoes import psd_variation, ranking_definition, ranking_statistic
from gravyflow.tests.test_stream_search import (
    FS, SPINNING, coloured_noise, exact_spectrum, optimal_snr, search, signal)
from gravyflow.tests.test_stream_vetoes import HEAVY

BANK = TemplateBank.from_masses(*HEAVY)
OTHER = TemplateBank.from_masses([60.0, 40.0], [45.0, 30.0], [0.0, 0.5], [0.0, 0.4])


def test_the_ranking_recomputes_exactly_across_seeds_and_files(tmp_path):
    # The ranking is computed from exactly the inputs stored (v in float32),
    # so recomputing it from them, or from a file of them, gives the ranking
    # at filtering time bit for bit (on one platform): 24 seeds, about 1300
    # triggers. (Ranked from float64 v and stored with float32 v, about 12% of
    # triggers differed by one float32 unit.)
    mf = search(BANK, snr_threshold=4.0)
    count = 0
    for seed in range(1, 25):
        out = mf.filter(coloured_noise(np.random.default_rng(seed), 200.0), 1000.0, "L1")
        recomputed = ranking_statistic(out.snr, out.reduced_chisq, out.sg_chisq, out.psd_variation)
        np.testing.assert_array_equal(recomputed, out.reweighted_snr, err_msg=f"seed {seed}")
        np.testing.assert_array_equal(out.ranked().reweighted_snr, out.reweighted_snr, err_msg=f"seed {seed}")
        out.save(tmp_path / f"{seed}.npz")
        back = DetectorTriggers.load(tmp_path / f"{seed}.npz", BANK)
        np.testing.assert_array_equal(back.reweighted_snr, out.reweighted_snr, err_msg=f"seed {seed}")
        count += len(out)
    assert count > 1000, count


def triggers(**overrides):
    values = dict(detector="L1", time=[1000.5, 1001.5], template=[0, 1], snr=[6.0, 7.0], reduced_chisq=[1.0, 2.0],
                  span=(1000.0, 1100.0), sg_chisq=[1.0, 5.0], psd_variation=[1.0, 1.2], gates=[1050.0],
                  bank=bank_identity(BANK))
    values.update(overrides)
    return DetectorTriggers(**values)


def test_the_ranking_is_derived_never_given_or_stored(tmp_path):
    # The ranking is a function of the stored inputs and the ranking's
    # definition, so it is neither an argument nor in the file: a file can
    # hold no ranking that disagrees with its inputs.
    with pytest.raises(TypeError, match="reweighted_snr"):
        triggers(reweighted_snr=[10.0, 10.0])
    out = triggers()
    np.testing.assert_array_equal(out.reweighted_snr, ranking_statistic(out.snr, out.reduced_chisq, out.sg_chisq,
                                                                        out.psd_variation))
    path = tmp_path / "triggers.npz"
    out.save(path)
    with np.load(path) as data:
        assert "reweighted_snr" not in data.files
        header = json.loads(str(data["header"]))
    # A definition with the sine-Gaussian veto off ranks the same inputs without it, on load.
    rewrite(path, {**header, "ranking": ranking_definition(sine_gaussian=False)})
    back = DetectorTriggers.load(path, BANK)
    np.testing.assert_array_equal(back.reweighted_snr, ranking_statistic(out.snr, out.reduced_chisq, None,
                                                                         out.psd_variation))
    assert back.reweighted_snr[1] > out.reweighted_snr[1]  # its sg_chisq of 5 no longer lowers it
    # The arrays are read-only, so the derived ranking cannot go stale.
    with pytest.raises(ValueError, match="read-only"):
        out.snr[0] = 100.0


def test_a_file_of_schema_version_2_with_a_stored_ranking_is_refused(tmp_path):
    path, header = stored(tmp_path)
    with np.load(path) as data:
        snr = data["snr"]
    rewrite(path, {**header, "schema_version": 2}, extra={"reweighted_snr": 2.0 * snr})
    with pytest.raises(ValueError, match="schema version 2"):
        DetectorTriggers.load(path, BANK)


def test_template_indices_outside_the_bank_are_refused(tmp_path):
    # BANK holds two templates: index 2 is outside it.
    for bad in (2, 999):
        with pytest.raises(ValueError, match="template"):
            triggers(template=[0, bad], bank=BANK)
        out = triggers(template=[0, bad])  # an identity alone cannot bound them
        out.save(tmp_path / "triggers.npz")
        with pytest.raises(ValueError, match="template"):
            DetectorTriggers.load(tmp_path / "triggers.npz", BANK)
    assert triggers(template=[0, 1], bank=BANK).bank == bank_identity(BANK)


def test_an_unknown_header_key_is_refused(tmp_path):
    path, header = stored(tmp_path)
    rewrite(path, {**header, "note": "anything"})
    with pytest.raises(ValueError, match="note"):
        DetectorTriggers.load(path, BANK)


def test_a_detector_gravyflow_does_not_know_is_refused(tmp_path):
    for name in ("X1", "l1", ""):
        with pytest.raises(ValueError, match="detector"):
            triggers(detector=name)
    path, header = stored(tmp_path)
    rewrite(path, {**header, "detector": "X1"})
    with pytest.raises(ValueError, match="X1"):
        DetectorTriggers.load(path, BANK)
    assert triggers(detector="V1").detector == "V1"


def test_a_file_holds_its_schema_ranking_and_bank_and_round_trips(tmp_path):
    out = search(BANK, snr_threshold=5.0).filter(coloured_noise(np.random.default_rng(3), 200.0), 1000.0, "L1")
    assert out.bank == bank_identity(BANK) and out.ranking == ranking_definition()
    path = tmp_path / "l1.npz"
    out.save(path)
    with np.load(path) as data:
        header = json.loads(str(data["header"]))
        assert "reweighted_snr" not in data.files
    assert header["schema_version"] == TRIGGER_SCHEMA_VERSION == 3
    assert set(header) == {"format", "schema_version", "detector", "bank", "ranking"}
    assert header["bank"] == bank_identity(BANK) and header["ranking"] == ranking_definition()
    for bank in (BANK, bank_identity(BANK)):
        back = DetectorTriggers.load(path, bank)
        for name in ("time", "template", "snr", "reduced_chisq", "reweighted_snr", "sg_chisq", "psd_variation",
                     "gates"):
            np.testing.assert_array_equal(getattr(back, name), getattr(out, name))
            assert getattr(back, name).dtype == getattr(out, name).dtype
        assert back.span == out.span and back.detector == "L1"
        assert back.bank == out.bank and back.ranking == out.ranking
    # A ranking recomputed without a correction names that in its definition, and survives the file.
    plain = out.ranked(sine_gaussian=False, psd_variation=False)
    assert plain.ranking == ranking_definition(sine_gaussian=False, psd_variation=False)
    plain.save(tmp_path / "plain.npz")
    assert DetectorTriggers.load(tmp_path / "plain.npz", BANK).ranking == plain.ranking


def test_the_bank_identity_binds_the_template_indices():
    assert bank_identity(BANK) == bank_identity(TemplateBank.from_masses(*HEAVY))
    assert bank_identity(BANK) != bank_identity(OTHER)  # one spin differs
    assert bank_identity(BANK) != bank_identity(TemplateBank.from_masses(*(values[:1] for values in HEAVY)))
    out = search(BANK, snr_threshold=5.0, psd_variation=None).filter(
        coloured_noise(np.random.default_rng(4), 200.0), 0.0, "L1")
    assert out.bank == bank_identity(BANK)
    assert out.ranking == ranking_definition(psd_variation=False)


def rewrite(path, header=None, drop=(), extra=None):
    """The stored file at `path` with its header replaced (a dict; None keeps it) and arrays dropped or added."""
    with np.load(path) as data:
        arrays = {name: data[name] for name in data.files if name not in drop}
    if header is not None:
        arrays["header"] = np.array(json.dumps(header))
    arrays.update(extra or {})
    np.savez(path, **arrays)


def stored(tmp_path):
    path = tmp_path / "triggers.npz"
    triggers().save(path)
    with np.load(path) as data:
        return path, json.loads(str(data["header"]))


@pytest.mark.parametrize("version", [1, 2, 4, "3"])
def test_a_file_of_another_schema_version_is_refused(tmp_path, version):
    path, header = stored(tmp_path)
    rewrite(path, {**header, "schema_version": version})
    with pytest.raises(ValueError, match="schema version"):
        DetectorTriggers.load(path, BANK)


def test_a_file_of_another_format_is_refused(tmp_path):
    path, header = stored(tmp_path)
    rewrite(path, {**header, "format": "something else"})
    with pytest.raises(ValueError, match="not a file of stored DetectorTriggers"):
        DetectorTriggers.load(path, BANK)


@pytest.mark.parametrize("change", [
    {"name": "newsnr"},
    {"parameters": {**ranking_definition()["parameters"], "sg_threshold": 6.0}},
    {"parameters": {**ranking_definition()["parameters"], "newsnr_q": 5.0}},
    {"parameters": {key: value for key, value in ranking_definition()["parameters"].items() if key != "psd_variation"}},
])
def test_a_file_of_another_ranking_is_refused(tmp_path, change):
    path, header = stored(tmp_path)
    rewrite(path, {**header, "ranking": {**header["ranking"], **change}})
    with pytest.raises(ValueError, match="ranking"):
        DetectorTriggers.load(path, BANK)


def test_a_file_of_another_bank_is_refused(tmp_path):
    path, _ = stored(tmp_path)
    with pytest.raises(ValueError, match="bank"):
        DetectorTriggers.load(path, OTHER)
    with pytest.raises(ValueError, match="bank"):
        DetectorTriggers.load(path, "not an identity")


def test_a_file_in_the_first_layout_is_refused_by_name(tmp_path):
    # Schema version 1 stored the arrays (the ranking among them), the detector, span and gates, with no header.
    path, _ = stored(tmp_path)
    with np.load(path) as data:
        snr = data["snr"]
    rewrite(path, drop=("header",), extra={"detector": np.array("L1"), "reweighted_snr": snr})
    with pytest.raises(ValueError, match="schema version 1"):
        DetectorTriggers.load(path, BANK)


def test_a_file_with_a_missing_extra_or_retyped_array_is_refused(tmp_path):
    path, header = stored(tmp_path)
    rewrite(path, drop=("sg_chisq",))
    with pytest.raises(ValueError, match="sg_chisq"):
        DetectorTriggers.load(path, BANK)
    path, header = stored(tmp_path)
    rewrite(path, extra={"stray": np.zeros(2)})
    with pytest.raises(ValueError, match="stray"):
        DetectorTriggers.load(path, BANK)
    path, header = stored(tmp_path)
    with np.load(path) as data:
        snr = data["snr"]
    rewrite(path, extra={"snr": snr.astype(np.float64)})
    with pytest.raises(ValueError, match="snr"):
        DetectorTriggers.load(path, BANK)


def test_a_file_that_is_not_an_npz_archive_is_refused(tmp_path):
    path = tmp_path / "triggers.npz"
    path.write_bytes(b"not an archive")
    with pytest.raises(ValueError, match="not a file of stored DetectorTriggers"):
        DetectorTriggers.load(path, BANK)
    with zipfile.ZipFile(tmp_path / "empty.npz", "w"):
        pass
    with pytest.raises(ValueError, match="not a file of stored DetectorTriggers"):
        DetectorTriggers.load(tmp_path / "empty.npz", BANK)


@pytest.mark.parametrize("name", ["snr", "time", "psd_variation", "sg_chisq", "reduced_chisq", "gates"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_a_non_finite_value_is_refused_on_construction(name, bad):
    values = {"time": [1000.5, 1001.5], "snr": [6.0, 7.0], "psd_variation": [1.0, 1.2], "sg_chisq": [1.0, 5.0],
              "reduced_chisq": [1.0, 2.0], "gates": [1050.0]}
    value = np.array(values[name], dtype=np.float64)
    value[-1] = bad
    with pytest.raises(ValueError, match=name):
        triggers(**{name: value})


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_a_non_finite_span_is_refused_on_construction(bad):
    with pytest.raises(ValueError, match="span"):
        triggers(span=(1000.0, bad))


@pytest.mark.parametrize("name", ["snr", "time", "psd_variation", "sg_chisq", "reduced_chisq", "gates"])
def test_a_non_finite_value_is_refused_on_save(tmp_path, name):
    # The arrays are read-only; one made writable and changed in place is still refused when saved.
    out = triggers()
    value = getattr(out, name)
    value.setflags(write=True)
    value[0] = np.nan
    with pytest.raises(ValueError, match=name):
        out.save(tmp_path / "triggers.npz")
    assert not (tmp_path / "triggers.npz").exists()


def test_triggers_need_a_bank_identity_and_a_known_ranking():
    with pytest.raises(ValueError, match="bank"):
        triggers(bank="")
    with pytest.raises(ValueError, match="ranking"):
        triggers(ranking={"name": "newsnr", "parameters": {}})


def test_triggers_of_different_banks_or_rankings_never_mix():
    one = triggers()
    other = triggers(bank=bank_identity(OTHER), detector="H1")
    with pytest.raises(ValueError, match="bank"):
        coincide({"L1": one, "H1": other})
    plain = triggers(detector="H1").ranked(sine_gaussian=False)
    with pytest.raises(ValueError, match="ranking"):
        coincide({"L1": one, "H1": plain})
    with pytest.raises(ValueError, match="bank"):
        DetectorTriggers.concatenate([one, triggers(bank=bank_identity(OTHER))])
    with pytest.raises(ValueError, match="ranking"):
        DetectorTriggers.concatenate([one, one.ranked(psd_variation=False)])
    with pytest.raises(ValueError, match="bank"):
        fit_templates(one, OTHER, threshold=5.0, lower_frequency_hertz=20.0)


def test_noise_free_strain_has_no_psd_variation_and_is_refused_by_name():
    # PSD variation compares noise with its own long-term spectrum; a lone
    # signal in no noise has a long-term (median) spectrum of zero, where it
    # is undefined. It is refused by name, never stored as NaN.
    seconds = 301.0
    strain = signal(SPINNING, [150.0], seconds)
    with pytest.raises(ValueError, match="PSD variation"):
        psd_variation(strain, FS)
    with pytest.raises(ValueError, match="PSD variation"):
        psd_variation(np.zeros(int(64 * FS)), FS)
    spectrum = exact_spectrum()
    strain *= 20.0 / optimal_snr(strain, spectrum)
    with pytest.raises(ValueError, match="PSD variation"):
        search(TemplateBank.from_masses(*SPINNING)).filter(strain, 0.0, "L1", psd=spectrum)
    # Without it, the same strain is searched.
    out = search(TemplateBank.from_masses(*SPINNING), psd_variation=None).filter(strain, 0.0, "L1", psd=spectrum)
    assert len(out) == 1 and out.snr[0] > 19.0 and np.all(out.psd_variation == 1.0)
