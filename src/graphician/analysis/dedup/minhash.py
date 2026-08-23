"""MinHash implementation for Jaccard similarity estimation."""

from __future__ import annotations


class MinHash:
    """MinHash signature for estimating Jaccard similarity.

    Uses splitmix64-derived permutation functions. Two sets produce the same
    minimum hash value with probability equal to their Jaccard similarity.
    """

    def __init__(self, num_permutations: int) -> None:
        self.num_permutations = num_permutations
        # Double hashing: (a * h1 + b) % p
        self.a: list[int] = []
        self.b: list[int] = []
        self.signature: list[int] = [2**32 - 1] * num_permutations
        self._init_permutations()

    def _splitmix64(self, state: list[int]) -> int:
        state[0] = (state[0] + 0x9e3779b97f4a7c15) & 0xFFFFFFFFFFFFFFFF
        z = state[0]
        z = ((z ^ (z >> 30)) * 0xbf58476d1ce4e5b9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94d049bb133111eb) & 0xFFFFFFFFFFFFFFFF
        return z ^ (z >> 31)

    def _init_permutations(self) -> None:
        state = [0x6c62272e07bb0142]
        seeds = []
        for _ in range(self.num_permutations):
            seeds.append(self._splitmix64(state))
            seeds.append(self._splitmix64(state))
        self.a = [(seeds[i] ^ i) for i in range(self.num_permutations)]
        self.b = [(seeds[i] ^ (i * 31)) for i in range(self.num_permutations)]

    @staticmethod
    def _hash_with_salt(data: bytes, salt: int) -> int:
        """FNV-1a hash with salt."""
        h = salt & 0xFFFFFFFFFFFFFFFF
        for byte in data:
            h = h ^ byte
            h = ((h * 0x00000100000001B3) & 0xFFFFFFFFFFFFFFFF)
        return h

    def update(self, shingle: str) -> None:
        """Update MinHash with a shingle."""
        data = shingle.encode("utf-8")
        for i in range(self.num_permutations):
            h1 = self._hash_with_salt(data, self.a[i])
            h2 = self._hash_with_salt(data, self.b[i])
            h = ((self.a[i] * (h1 % 65521) + self.b[i] + h2) % 65521) & 0xFFFFFFFF
            if h < self.signature[i]:
                self.signature[i] = h

    @classmethod
    def from_iter(cls, items: list[str], num_permutations: int) -> MinHash:
        """Build MinHash from an iterable of strings."""
        mh = cls(num_permutations)
        for item in items:
            mh.update(item)
        return mh

    def jaccard(self, other: MinHash) -> float:
        """Estimate Jaccard similarity between two signatures."""
        if not self.signature:
            return 0.0
        matches = sum(1 for a, b in zip(self.signature, other.signature) if a == b)
        return matches / len(self.signature)


def shingle(label: str, size: int) -> list[str]:
    """Generate character n-gram shingles from a label."""
    if len(label) < size:
        return [label]
    return [label[i:i + size] for i in range(len(label) - size + 1)]
