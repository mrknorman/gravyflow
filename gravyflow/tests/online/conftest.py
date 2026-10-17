"""
Live-source conformance: the public sources gravyflow reads (GWOSC's run, timeline
and strain APIs, and the Gravity Spy release on Zenodo) still say what gravyflow's
committed tables and offline fakes say they do.

Everything gravyflow itself does with these sources is tested offline, on committed
fakes, in the default suites; this package checks only that the sources have not
drifted under those fakes. It is kept out of the default suites and run on its own,
with ``--network`` (without it the suite refuses every connection):

    pytest --network gravyflow/tests/online

The hosts are probed once per run. A host that cannot be reached fails every test,
naming it (``ONLINE REFUSED: <host> unreachable``): an offline run is no verdict on
the sources, and is never a pass or a skip.
"""

from __future__ import annotations

import socket
from typing import Dict, Optional

import pytest

#: Every host this package reads, probed once per run.
HOSTS = ("gwosc.org", "zenodo.org")
#: Seconds a probe waits for a TCP connection to a host's HTTPS port.
PROBE_SECONDS = 10.0


def probe(host: str) -> Optional[str]:
    """None when ``host`` accepts a TCP connection on 443; else why not."""
    try:
        socket.create_connection((host, 443), timeout=PROBE_SECONDS).close()
    except OSError as error:
        return f"{type(error).__name__}: {error}"
    return None


@pytest.fixture(scope="session")
def unreachable(suite_guard) -> Dict[str, str]:
    """{host: why it cannot be reached}, probed once; empty when every host answers."""
    if suite_guard.offline:
        # The suite refuses the network without --network: no probe leaves this machine.
        return {host: "this run refuses the network (pass --network)" for host in HOSTS}
    return {host: why for host in HOSTS if (why := probe(host)) is not None}


@pytest.fixture(autouse=True)
def online(unreachable) -> Dict[str, str]:
    """Every test here reads the probe; ``pytest_pyfunc_call`` acts on it."""
    return unreachable


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    """A test fails (in its call, never skipped), naming the host, when a host cannot be reached."""
    unreachable = pyfuncitem.funcargs["online"]
    if unreachable:
        host, why = next(iter(unreachable.items()))
        pytest.fail(f"ONLINE REFUSED: {host} unreachable ({why})", pytrace=False)
