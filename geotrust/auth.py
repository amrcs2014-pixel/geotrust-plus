"""TESLA-in-ranging (GeoTrust layer 1): one-way key chain + truncated AES-CMAC per ranging unit.

Round r: RU_r carries MAC_{K_r}(RU_r) (8 B) and discloses K_{r-1} (16 B)  -> 24 B per RU.
A receiver buffers RU_r, and authenticates it at round r+1 when K_r is disclosed and checked
against the chain commitment (K_{r} = H(K_{r+1}),  K_0 signed offline).
"""
import hashlib
import os
import time
from cryptography.hazmat.primitives.cmac import CMAC
from cryptography.hazmat.primitives.ciphers import algorithms

KEY_B, MAC_B = 16, 8


def H(k):
    return hashlib.sha256(k).digest()[:KEY_B]


class Chain:
    def __init__(self, length=2 ** 16, seed=None):
        k = seed or os.urandom(KEY_B)
        self.keys = [k]
        for _ in range(length):
            self.keys.append(H(self.keys[-1]))
        self.keys.reverse()                         # keys[0] = commitment K_0
        self.L = length

    def key(self, r):
        return self.keys[r]


def mac(key, msg):
    c = CMAC(algorithms.AES(key))
    c.update(msg)
    return c.finalize()[:MAC_B]


def frame(chain, r, payload):
    return payload + mac(chain.key(r), payload) + (chain.key(r - 1) if r > 0 else bytes(KEY_B))


class Receiver:
    def __init__(self, commitment):
        self.last_key, self.last_r = commitment, 0
        self.buf = {}

    def _check_key(self, k, r):
        x, steps = k, r - self.last_r
        if steps <= 0:
            return False
        for _ in range(steps):
            x = H(x)
        return x == self.last_key

    def receive(self, r, fr):
        """Returns the payloads authenticated at this round (those of round r-1)."""
        payload, m, kd = fr[:-MAC_B - KEY_B], fr[-MAC_B - KEY_B:-KEY_B], fr[-KEY_B:]
        self.buf[r] = (payload, m)
        ok = []
        if r > 0 and self._check_key(kd, r - 1):
            self.last_key, self.last_r = kd, r - 1
            if (r - 1) in self.buf:
                p, mm = self.buf.pop(r - 1)
                if mac(kd, p) == mm:
                    ok.append(p)
        return ok


def ru_bytes(k_nbrs):
    """Swarm-ranging RU size: header (id 2, seq 2, Tx ts 5) + per neighbour
    (id 2, Rx ts 5, distance 2, PLR 1, flags 1)."""
    return 9 + 11 * k_nbrs


def overhead_table(ks=(5, 10, 20, 30, 40)):
    return [{"neighbours": k, "RU_bytes": ru_bytes(k), "auth_bytes": MAC_B + KEY_B,
             "overhead_pct": 100 * (MAC_B + KEY_B) / ru_bytes(k)} for k in ks]


def bench_cmac(n=20000, k=20):
    key, msg = os.urandom(16), os.urandom(ru_bytes(k))
    t0 = time.perf_counter()
    for _ in range(n):
        mac(key, msg)
    return (time.perf_counter() - t0) / n * 1e6    # µs per MAC on this host CPU
