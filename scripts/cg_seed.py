"""Validated runtime seeding for the vendored cabt engine.

The public ``BattleStart`` API has no seed argument.  The Windows engine used
by this repository nevertheless contains a per-battle ``std::mt19937`` and an
option byte that chooses it instead of ``std::random_device``.  This module
changes those two constructor instructions *in process*; it never edits the
DLL on disk or any agent.

This is deliberately fail-closed.  A patch is only attempted for a known DLL
hash, from the repository's ``vendor/cg`` directory, with the exact expected
machine code at both sites.  A different engine build raises
``UnsupportedEngineError`` instead of writing to an assumed address.

The patch only makes the engine seed-addressable.  Callers must still run the
reproducibility probe in ``scripts/probe_cg_seed.py`` before treating an engine
build as suitable for common-random-number evaluation.
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wintypes
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path


class UnsupportedEngineError(RuntimeError):
    """The loaded native engine is not a build this module can safely patch."""


@dataclass(frozen=True)
class PatchLayout:
    sha256: str
    seed_rva: int
    flag_rva: int
    seed_original: bytes
    flag_original: bytes


VENDORED_WINDOWS_LAYOUT = PatchLayout(
    sha256="e758cdba51482102ac1463e58b621951431d6d5b0ce46b8facee65cae2ee17f9",
    seed_rva=0x2A0DB,
    flag_rva=0x2A11A,
    seed_original=bytes.fromhex("8b8f20020000"),
    flag_original=bytes.fromhex("0fb6872b020000"),
)

_FLAG_PATCH = bytes.fromhex("31c09090909090")
_PAGE_EXECUTE_READWRITE = 0x40


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class EngineSeedController:
    """Own and restore the two in-memory instructions used for engine seeding."""

    def __init__(self) -> None:
        if os.name != "nt":
            raise UnsupportedEngineError(
                "native seeding is currently verified only for the vendored Windows cg.dll"
            )

        # Import lazily so scripts can put ROOT/vendor first on sys.path before
        # constructing the controller.
        from cg import sim  # type: ignore

        engine_path = Path(sim.lib_path).resolve()
        expected_dir = (Path(__file__).resolve().parents[1] / "vendor" / "cg").resolve()
        if engine_path.parent != expected_dir:
            raise UnsupportedEngineError(
                f"wrong cg engine loaded: {engine_path}; expected one under {expected_dir}"
            )

        layout = VENDORED_WINDOWS_LAYOUT
        actual_hash = _sha256(engine_path)
        if actual_hash != layout.sha256:
            raise UnsupportedEngineError(
                f"unsupported cg.dll sha256={actual_hash}; expected {layout.sha256}"
            )

        self.engine_path = engine_path
        self.layout = layout
        self._lib = sim.lib
        self._base = ctypes.cast(self._lib._handle, ctypes.c_void_p).value
        if not self._base:
            raise UnsupportedEngineError("could not resolve the loaded cg.dll base address")
        self._installed = False
        self._verify_original_or_installed()

    def _read(self, rva: int, size: int) -> bytes:
        return ctypes.string_at(self._base + rva, size)

    def _write(self, rva: int, data: bytes) -> None:
        address = self._base + rva
        old = wintypes.DWORD()
        kernel32 = ctypes.windll.kernel32
        ok = kernel32.VirtualProtect(
            ctypes.c_void_p(address),
            ctypes.c_size_t(len(data)),
            _PAGE_EXECUTE_READWRITE,
            ctypes.byref(old),
        )
        if not ok:
            raise OSError("VirtualProtect failed while enabling the cg seed patch")
        try:
            ctypes.memmove(ctypes.c_void_p(address), data, len(data))
            kernel32.FlushInstructionCache(
                ctypes.c_void_p(-1), ctypes.c_void_p(address), ctypes.c_size_t(len(data))
            )
        finally:
            previous = wintypes.DWORD()
            kernel32.VirtualProtect(
                ctypes.c_void_p(address),
                ctypes.c_size_t(len(data)),
                old.value,
                ctypes.byref(previous),
            )

    @staticmethod
    def _seed_patch(seed: int) -> bytes:
        seed &= 0xFFFFFFFF
        if seed == 0:
            seed = 1
        return b"\xb9" + seed.to_bytes(4, "little") + b"\x90"

    def _verify_original_or_installed(self) -> None:
        seed_bytes = self._read(self.layout.seed_rva, len(self.layout.seed_original))
        flag_bytes = self._read(self.layout.flag_rva, len(self.layout.flag_original))
        seed_ok = seed_bytes == self.layout.seed_original or seed_bytes[:1] == b"\xb9"
        flag_ok = flag_bytes in (self.layout.flag_original, _FLAG_PATCH)
        if not seed_ok or not flag_ok:
            raise UnsupportedEngineError(
                "cg.dll instruction signatures do not match the verified layout: "
                f"seed={seed_bytes.hex()} flag={flag_bytes.hex()}"
            )

    def install(self) -> None:
        """Select the battle-local MT instead of ``random_device``."""
        self._verify_original_or_installed()
        flag_bytes = self._read(self.layout.flag_rva, len(self.layout.flag_original))
        if flag_bytes == self.layout.flag_original:
            self._write(self.layout.flag_rva, _FLAG_PATCH)
        self._installed = True

    def set_seed(self, seed: int) -> None:
        """Set the nonzero 32-bit master seed used by the next battle."""
        self.install()
        self._write(self.layout.seed_rva, self._seed_patch(seed))

    def restore(self) -> None:
        """Restore the original instructions.  Safe to call more than once."""
        self._verify_original_or_installed()
        self._write(self.layout.seed_rva, self.layout.seed_original)
        self._write(self.layout.flag_rva, self.layout.flag_original)
        self._installed = False

    def status(self) -> dict[str, object]:
        return {
            "engine_path": str(self.engine_path),
            "engine_sha256": self.layout.sha256,
            "installed": self._installed,
            "seed_instruction": self._read(
                self.layout.seed_rva, len(self.layout.seed_original)
            ).hex(),
            "flag_instruction": self._read(
                self.layout.flag_rva, len(self.layout.flag_original)
            ).hex(),
        }

    def __enter__(self) -> "EngineSeedController":
        self.install()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.restore()

