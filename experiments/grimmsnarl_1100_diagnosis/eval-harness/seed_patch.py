"""Make vendor/cg's native engine seedable, in-process, without touching disk.

Findings from vendor/cg/cg.dll (Windows) and vendor/cg/libcg.so (Linux):

  The battle object ("ApiData", 0x7058 bytes on Windows) carries its own
  std::mt19937:
      ApiData+0x228  uint32 seed
      ApiData+0x230..0x233  four option bytes
      ApiData+0x238  uint32 mt19937 _Idx
      ApiData+0x23c  uint32 mt19937 state[624]      (state[0] == seed)

  ShuffleDeck(State&,int,bool) and SelectCoinSingle(State&,int) both do
      cmp byte [ApiData+0x233], 0
      jne  <use a fresh std::random_device("default")>
      ...  <else use the ApiData mt19937>
  so a single byte selects "OS entropy" vs "seeded MT".

  The battle-construction helper at RVA 0x29fd0 copies an option block into
  ApiData:
      1800 2a0db  8b 8f 20 02 00 00     mov  ecx,[rdi+0x220]   ; opt.seed
      1800 2a0e1  89 8b 28 02 00 00     mov  [rbx+0x228],ecx
      ...
      1800 2a11a  0f b6 87 2b 02 00 00  movzx eax,byte [rdi+0x22b]
      1800 2a121  88 83 33 02 00 00     mov  [rbx+0x233],al    ; the flag
      1800 2a127  85 c9                 test ecx,ecx
      1800 2a129  75 0d                 jne  +0xd              ; seed!=0 -> keep
      1800 2a12b  e8 ..                 call <random_device>   ; else draw one
      1800 2a132  89 83 28 02 00 00     mov  [rbx+0x228],eax
      1800 2a138  ...                   seed the mt19937 from ecx

  The shipped callers always pass opt.seed = 0 and opt.flag = 1, i.e. "draw a
  fresh OS seed AND ignore it, shuffle from random_device". Two immediates are
  all that stand between us and common random numbers:

    A) RVA 0x2a0db  mov ecx,[rdi+0x220]   -> mov ecx, <SEED>        ; b9 imm32 90
    B) RVA 0x2a11a  movzx eax,[rdi+0x22b] -> xor eax,eax + nops     ; 31 c0 90*5

  Both are 100% inside the engine. No agent code, no deck code, no api shim.
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "vendor") not in sys.path:
    sys.path.insert(0, str(ROOT / "vendor"))

from cg.sim import lib  # noqa: E402

RVA_SEED = 0x2A0DB
RVA_FLAG = 0x2A11A
ORIG_SEED = bytes.fromhex("8b8f20020000")
ORIG_FLAG = bytes.fromhex("0fb6872b020000")
PATCH_FLAG = bytes.fromhex("31c09090909090")

PAGE_EXECUTE_READWRITE = 0x40
_BASE = ctypes.cast(lib._handle, ctypes.c_void_p).value


def _unprotect(addr: int, size: int) -> int:
    old = wt.DWORD()
    ok = ctypes.windll.kernel32.VirtualProtect(
        ctypes.c_void_p(addr), ctypes.c_size_t(size),
        PAGE_EXECUTE_READWRITE, ctypes.byref(old))
    if not ok:
        raise OSError("VirtualProtect failed")
    return old.value


def _reprotect(addr: int, size: int, old: int) -> None:
    prev = wt.DWORD()
    ctypes.windll.kernel32.VirtualProtect(
        ctypes.c_void_p(addr), ctypes.c_size_t(size),
        old, ctypes.byref(prev))


def _read(addr: int, size: int) -> bytes:
    return ctypes.string_at(addr, size)


def _write(addr: int, data: bytes) -> None:
    old = _unprotect(addr, len(data))
    ctypes.memmove(ctypes.c_void_p(addr), data, len(data))
    _reprotect(addr, len(data), old)
    ctypes.windll.kernel32.FlushInstructionCache(
        ctypes.c_void_p(-1), ctypes.c_void_p(addr), ctypes.c_size_t(len(data)))


_installed = False


def verify() -> dict:
    return {
        "base": hex(_BASE),
        "seed_site": _read(_BASE + RVA_SEED, 6).hex(),
        "flag_site": _read(_BASE + RVA_FLAG, 7).hex(),
    }


def install() -> None:
    """Disable the random_device path. Idempotent."""
    global _installed
    if _installed:
        return
    cur = _read(_BASE + RVA_FLAG, 7)
    if cur != ORIG_FLAG and cur != PATCH_FLAG:
        raise RuntimeError(f"unexpected bytes at flag site: {cur.hex()}")
    if _read(_BASE + RVA_SEED, 6)[:1] not in (b"\x8b", b"\xb9"):
        raise RuntimeError("unexpected bytes at seed site")
    _write(_BASE + RVA_FLAG, PATCH_FLAG)
    _installed = True


def set_seed(seed: int) -> None:
    """Force the next battle_start()'s master seed. seed must be nonzero."""
    install()
    seed &= 0xFFFFFFFF
    if seed == 0:
        seed = 1
    _write(_BASE + RVA_SEED,
           b"\xb9" + seed.to_bytes(4, "little") + b"\x90")


def uninstall() -> None:
    global _installed
    _write(_BASE + RVA_SEED, ORIG_SEED)
    _write(_BASE + RVA_FLAG, ORIG_FLAG)
    _installed = False
