import threading

import numpy as np

from kinetixrt.core.action_buffer import ActionBuffer
from kinetixrt.types import ActionChunk, ComputeBudget


def chunk(seq, obs_tick, n=10, value=None, anchor=None):
    actions = np.full((n, 2), float(seq) if value is None else value, dtype=np.float32)
    actions[:, 1] = np.arange(n)
    return ActionChunk(seq, seq, obs_tick, 0.0, 0.0, 0.0, 0.0, actions, ComputeBudget(10), 0.0, anchor_tick=anchor)


def test_tick_alignment_skips_late_actions():
    buf = ActionBuffer()
    rep = buf.install(chunk(0, obs_tick=5), now_tick=8)
    assert rep.accepted and rep.skipped_actions == 3
    a, c = buf.get(8)
    assert a[1] == 3 and c.seq == 0
    assert buf.get(14)[0][1] == 9
    assert buf.get(15)[0] is None  # plan ran out
    assert buf.status(8).remaining == 7 and buf.status(8).executed == 3


def test_anchor_moves_alignment_without_changing_observation_tick():
    buf = ActionBuffer()
    assert buf.install(chunk(0, obs_tick=0, anchor=4), now_tick=4).skipped_actions == 0
    a, c = buf.get(4)
    assert a[1] == 0 and c.obs_tick == 0


def test_rejects_superseded_expired_and_invalidated_chunks():
    buf = ActionBuffer()
    assert buf.install(chunk(1, obs_tick=10), now_tick=10).accepted
    assert buf.install(chunk(2, obs_tick=9), now_tick=11).reason == "superseded_by_newer_chunk"
    assert buf.install(chunk(3, obs_tick=11, n=4), now_tick=20).reason == "all_actions_expired"
    buf.invalidate_before(30)
    assert buf.install(chunk(4, obs_tick=25), now_tick=26).reason == "invalidated_observation"
    assert buf.install(chunk(5, obs_tick=30), now_tick=31).accepted
    assert buf.rejected_stale == 3 and buf.accepted == 2


def test_disagreement_is_measured_on_overlap():
    buf = ActionBuffer(disagreement_dims=(0,))
    buf.install(chunk(0, obs_tick=0, n=20, value=0.0), now_tick=0)
    rep = buf.install(chunk(1, obs_tick=5, n=20, value=0.5), now_tick=5)
    assert abs(rep.disagreement - 0.5) < 1e-6


def test_blending_interpolates_between_plans():
    buf = ActionBuffer(blend_ticks=3)
    buf.install(chunk(0, obs_tick=0, n=20, value=0.0), now_tick=0)
    buf.install(chunk(1, obs_tick=5, n=20, value=1.0), now_tick=5)
    vals = [buf.get(t)[0][0] for t in range(5, 9)]
    assert vals[0] < vals[1] < vals[2] <= vals[3] == 1.0


def test_concurrent_install_and_get_never_returns_torn_actions():
    buf = ActionBuffer()
    stop = threading.Event()
    errors = []

    def writer():
        seq = 0
        while not stop.is_set():
            buf.install(chunk(seq, obs_tick=seq, n=50, value=float(seq)), now_tick=seq)
            seq += 1

    def reader():
        while not stop.is_set():
            st = buf.status(0)
            for t in range(st.obs_tick or 0, (st.obs_tick or 0) + 5):
                a, c = buf.get(t)
                if a is not None and a[0] != float(c.seq):
                    errors.append((a[0], c.seq))

    threads = [threading.Thread(target=writer), threading.Thread(target=reader), threading.Thread(target=reader)]
    for t in threads:
        t.start()
    threading.Event().wait(0.3)
    stop.set()
    for t in threads:
        t.join()
    assert not errors
