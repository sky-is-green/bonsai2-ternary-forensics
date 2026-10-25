#!/usr/bin/env python3
"""CPU-only PTQ1_0 vs PQ2_0 packing parity for Bonsai 2 27B.

Reads both GGUF files, decodes ternary blocks, and compares the trits and
fp16 group scales.  PTQ1_0 decoder is a port of strata/kernels/cpu/ptq1_0.cpp;
PQ2_0 decoder mirrors bonsai_forensics/pq2_0.py.
"""
from __future__ import annotations
import argparse
import struct
from pathlib import Path

import numpy as np

ALIGN = 32
T_PQ2 = 142
T_PTQ1 = 143
POW3 = np.array([1, 3, 9, 27, 81], dtype=np.uint16)


class Header:
    def __init__(self, path: Path):
        self.path = path
        self.blob = path.read_bytes()
        if self.blob[:4] != b'GGUF':
            raise SystemExit(f'not GGUF: {path}')
        self.version, n_t, n_kv = struct.unpack_from('<IQQ', self.blob, 4)
        if self.version != 3:
            raise SystemExit(f'version {self.version}')
        off = 24
        self.meta = {}
        for _ in range(n_kv):
            k, off = self._str(off)
            t, = struct.unpack_from('<I', self.blob, off); off += 4
            v, off = self._val(off, t)
            self.meta[k] = v
        self.tensors = {}
        for _ in range(n_t):
            name, off = self._str(off)
            nd, = struct.unpack_from('<I', self.blob, off); off += 4
            shape = struct.unpack_from('<' + 'Q'*nd, self.blob, off); off += 8*nd
            typ, toff = struct.unpack_from('<IQ', self.blob, off); off += 12
            self.tensors[name] = (tuple(int(x) for x in shape), int(typ), int(toff))
        self.data_start = (off + ALIGN - 1) // ALIGN * ALIGN

    def _str(self, off):
        n, = struct.unpack_from('<Q', self.blob, off); off += 8
        return self.blob[off:off+n].decode('utf-8'), off+n

    def _val(self, off, typ):
        # only the metadata types present in these files
        if typ == 8: return self._str(off)
        if typ == 9:
            et, n = struct.unpack_from('<IQ', self.blob, off); off += 12
            vals = []
            for _ in range(n):
                v, off = self._val(off, et); vals.append(v)
            return vals, off
        fmt, size = {0:('<B',1),1:('<b',1),2:('<H',2),3:('<h',2),4:('<I',4),5:('<i',4),6:('<f',4),7:('<B',1),10:('<Q',8),11:('<q',8),12:('<d',8)}[typ]
        v, = struct.unpack_from(fmt, self.blob, off)
        return v, off+size

    def nbytes(self, shape, typ):
        ne0 = int(shape[0])
        rows = int(np.prod(shape[1:], dtype=np.int64)) if len(shape) > 1 else 1
        if typ == T_PTQ1:
            return rows * (ne0 // 128) * 28
        if typ == T_PQ2:
            return rows * (ne0 // 128) * 34
        if typ == 0: return rows * ne0 * 4
        if typ == 1: return rows * ne0 * 2
        raise ValueError(f'unsupported type {typ} for {shape}')

    def read_rows(self, name, rows=None):
        shape, typ, toff = self.tensors[name]
        if typ not in (T_PTQ1, T_PQ2):
            raise ValueError(f'{name}: type {typ}')
        ne0 = shape[0]
        total_rows = int(np.prod(shape[1:], dtype=np.int64)) if len(shape) > 1 else 1
        nrows = total_rows if rows is None else min(rows, total_rows)
        rowbytes = (ne0 // 128) * (28 if typ == T_PTQ1 else 34)
        start = self.data_start + toff
        raw = self.blob[start:start + nrows * rowbytes]
        if typ == T_PTQ1:
            return self._ptq1(raw, nrows, ne0 // 128)
        return self._pq2(raw, nrows, ne0 // 128)

    @staticmethod
    def _ptq1(raw, rows, nblocks):
        b = np.frombuffer(raw, dtype=np.uint8).reshape(rows, nblocks, 28)
        code = np.empty((rows, nblocks, 128), dtype=np.int8)
        qs = b[:, :, :16]
        for n in range(5):
            q = (qs.astype(np.uint16) * POW3[n]) & 0xFF
            xi = (q.astype(np.uint16) * 3) >> 8
            code[:, :, n*16:n*16+16] = xi.astype(np.int8) - 1
        qs = b[:, :, 16:24]
        for n in range(5):
            q = (qs.astype(np.uint16) * POW3[n]) & 0xFF
            xi = (q.astype(np.uint16) * 3) >> 8
            code[:, :, 80+n*8:80+n*8+8] = xi.astype(np.int8) - 1
        qh = b[:, :, 24:26]
        for n in range(4):
            for h in range(2):
                q = (qh[:, :, h].astype(np.uint16) * POW3[n]) & 0xFF
                xi = (q.astype(np.uint16) * 3) >> 8
                code[:, :, 120+n*2+h] = xi.astype(np.int8) - 1
        scales = b[:, :, 26:28].copy().view('<f2').astype(np.float32)[:, :, 0]
        return code.reshape(rows, nblocks*128), scales

    @staticmethod
    def _pq2(raw, rows, nblocks):
        b = np.frombuffer(raw, dtype=np.uint8).reshape(rows, nblocks, 34)
        scales = b[:, :, :2].copy().view('<f2').astype(np.float32)[:, :, 0]
        qs = b[:, :, 2:].reshape(rows, nblocks, 32, 1)
        shifts = (2 * np.arange(4, dtype=np.uint8)).reshape(1, 1, 1, 4)
        q = ((qs >> shifts) & 3).astype(np.int8).reshape(rows, nblocks, 128)
        return (q - 1).reshape(rows, nblocks*128), scales


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ptq1', required=True)
    ap.add_argument('--pq2', required=True)
    ap.add_argument('--rows', type=int, default=4, help='rows per tensor; 0 = all')
    ap.add_argument('--all-tensors', action='store_true')
    ap.add_argument('--list', action='store_true')
    a = ap.parse_args()
    p = Header(Path(a.ptq1)); q = Header(Path(a.pq2))
    if a.list:
        for n in p.tensors:
            print(n, p.tensors[n][0], p.tensors[n][1])
        return
    names = [n for n, (_, typ, _) in p.tensors.items() if typ == T_PTQ1]
    if not a.all_tensors:
        # a few representative tensors: embedding, an attention projection, an FFN
        names = [n for n in ('token_embd.weight', 'output.weight',
                             'blk.0.attn_q.weight', 'blk.0.attn_k.weight',
                             'blk.0.ffn_gate.weight', 'blk.0.ffn_down.weight')
                 if n in p.tensors]
    if not names:
        raise SystemExit('no PTQ1_0 tensors selected')
    total_rows = total_bad = 0
    worst_scale = 0.0
    for name in names:
        rows = None if a.rows == 0 else a.rows
        pc, ps = p.read_rows(name, rows)
        qc, qs = q.read_rows(name, rows)
        if pc.shape != qc.shape:
            raise SystemExit(f'{name}: shape mismatch {pc.shape} vs {qc.shape}')
        bad = int((pc != qc).sum())
        scale_diff = np.abs(ps - qs)
        worst_scale = max(worst_scale, float(scale_diff.max()) if scale_diff.size else 0.0)
        total_rows += pc.shape[0]
        total_bad += bad
        print(f'{name:28} rows={pc.shape[0]:5} bad_codes={bad:10} max_scale_diff={scale_diff.max() if scale_diff.size else 0:.3e}')
    print(f'checked {len(names)} tensors / {total_rows} rows; bad_codes={total_bad}; worst_scale_diff={worst_scale:.3e}')
    print(f'ptq1={p.path.stat().st_size} bytes pq2={q.path.stat().st_size} bytes')

if __name__ == '__main__':
    main()
