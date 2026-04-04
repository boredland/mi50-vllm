"""
Patch for vllm's vision encoder SDPA to avoid 256 GB OOM on AMD MI50
(gfx906) caused by F.scaled_dot_product_attention falling back to naive
O(n^2) attention on ROCm without FlashAttention support.

Patches vit_attn_wrappers.py (v0.11.2) or qwen2_5_vl.py (v0.11.0),
replacing F.scaled_dot_product_attention() with a chunked implementation
that tiles along the query sequence dimension.

Run inside the container before starting vllm:
    python3 /patch/patch_qwen2_5_vl_attention.py

Idempotent -- running it twice is safe.
"""

import re
import site
import sys
from pathlib import Path

PATCH_MARKER = "CHUNKED SDPA PATCH"

CHUNKED_SDPA_CODE = '''
# --- BEGIN CHUNKED SDPA PATCH (gfx906 OOM fix) ---
import math as _math

def _chunked_sdpa(q, k, v, chunk_q=1024, dropout_p=0.0, scale=None, enable_gqa=False, **kwargs):
    """
    Drop-in replacement for F.scaled_dot_product_attention(q, k, v).
    Tiles over the query dimension to avoid O(seq^2) memory.
    Shape: (batch, heads, seq_len, head_dim) -> same.
    """
    import torch as _torch
    B, H, S, D = q.shape
    _scale = scale if scale is not None else 1.0 / _math.sqrt(D)
    if S <= chunk_q:
        # Small sequences are fine with naive attention
        scores = _torch.matmul(q, k.transpose(-2, -1)) * _scale
        attn = _torch.softmax(scores, dim=-1)
        if dropout_p > 0.0:
            attn = F.dropout(attn, p=dropout_p, training=False)
        return _torch.matmul(attn, v)
    out = _torch.empty_like(q)
    for start in range(0, S, chunk_q):
        end = min(start + chunk_q, S)
        q_c = q[:, :, start:end, :]
        scores = _torch.matmul(q_c, k.transpose(-2, -1)) * _scale
        attn = _torch.softmax(scores, dim=-1)
        if dropout_p > 0.0:
            attn = F.dropout(attn, p=dropout_p, training=False)
        out[:, :, start:end, :] = _torch.matmul(attn, v)
        del scores, attn
    return out
# --- END CHUNKED SDPA PATCH ---
'''


def find_file(filename):
    """Search for a file in vllm package directories."""
    candidates = []
    for base in site.getsitepackages() + [site.getusersitepackages()]:
        for p in Path(base).rglob(filename):
            candidates.append(p)
    for p in Path("/opt").rglob(f"vllm/**/{filename}"):
        candidates.append(p)
    for p in Path("/usr/lib").rglob(f"vllm/**/{filename}"):
        candidates.append(p)
    seen = set()
    for c in candidates:
        if c.exists() and str(c) not in seen:
            seen.add(str(c))
            return c
    return None


def patch_file(filepath):
    """Patch a single file, replacing F.scaled_dot_product_attention with _chunked_sdpa."""
    content = filepath.read_text()

    if PATCH_MARKER in content:
        print(f"[patch] {filepath.name}: already patched -- skipping.")
        return True

    if "F.scaled_dot_product_attention" not in content:
        return False

    # Insert chunked SDPA function at module level (after imports, before classes)
    # Find the last top-level import line
    import_pattern = r'^(?:import |from )\S+'
    last_import_end = 0
    for m in re.finditer(import_pattern, content, re.MULTILINE):
        last_import_end = m.end()
    if last_import_end > 0:
        # Find end of that line
        insert_pos = content.find("\n", last_import_end) + 1
    else:
        idx = content.find("F.scaled_dot_product_attention")
        insert_pos = content.rfind("\n", 0, idx) + 1

    content = content[:insert_pos] + CHUNKED_SDPA_CODE + "\n" + content[insert_pos:]

    # Replace F.scaled_dot_product_attention calls
    # Match the function call including multiline args, using balanced parens
    count = 0
    while "F.scaled_dot_product_attention(" in content:
        idx = content.find("F.scaled_dot_product_attention(")
        # Find matching closing paren
        start = idx + len("F.scaled_dot_product_attention(")
        depth = 1
        pos = start
        while pos < len(content) and depth > 0:
            if content[pos] == '(':
                depth += 1
            elif content[pos] == ')':
                depth -= 1
            pos += 1
        args = content[start:pos-1].strip()
        replacement = f'_chunked_sdpa({args})'
        content = content[:idx] + replacement + content[pos:]
        count += 1
    new_content = content

    if count == 0:
        print(f"[patch] {filepath.name}: WARNING - found F.scaled_dot_product_attention but regex didn't match.")
        return False

    # Backup and write
    backup = filepath.with_suffix(".py.bak")
    if not backup.exists():
        filepath.rename(backup)
    else:
        filepath.unlink()
    filepath.write_text(new_content)
    print(f"[patch] {filepath.name}: replaced {count} SDPA call(s) with _chunked_sdpa (chunk_q=1024)")
    return True


def patch_safetensors_cpu_loading():
    """
    Patch safetensors weight loading to use CPU then transfer to GPU.
    Fixes 'Memory access fault' on gfx906 with direct-to-GPU safetensors loading.
    """
    weight_utils = find_file("weight_utils.py")
    if not weight_utils:
        print("[patch] weight_utils.py not found - skipping CPU load patch.")
        return

    content = weight_utils.read_text()
    marker = "CPU LOAD PATCH"

    if marker in content:
        print(f"[patch] weight_utils.py: CPU load already patched -- skipping.")
        return

    # Patch safe_open get_tensor to load via CPU
    old = '''            with safe_open(st_file, framework="pt") as f:
                for name in f.keys():  # noqa: SIM118
                    param = f.get_tensor(name)
                    yield name, param'''

    new = '''            # --- BEGIN CPU LOAD PATCH (gfx906 memory fault fix) ---
            with safe_open(st_file, framework="pt", device="cpu") as f:
                for name in f.keys():  # noqa: SIM118
                    param = f.get_tensor(name)
                    yield name, param
            # --- END CPU LOAD PATCH ---'''

    if old in content:
        content = content.replace(old, new)
        backup = weight_utils.with_suffix(".py.bak")
        if not backup.exists():
            weight_utils.rename(backup)
        else:
            weight_utils.unlink()
        weight_utils.write_text(content)
        print(f"[patch] weight_utils.py: patched safetensors to load via CPU")
    else:
        print(f"[patch] weight_utils.py: could not find safe_open block to patch")


def main():
    patched = False

    # Patch safetensors CPU loading (gfx906 memory fault fix)
    patch_safetensors_cpu_loading()

    # v0.11.2: SDPA is in vit_attn_wrappers.py
    wrappers = find_file("vit_attn_wrappers.py")
    if wrappers:
        print(f"[patch] Found: {wrappers}")
        if patch_file(wrappers):
            patched = True

    # v0.11.0: SDPA is in qwen2_5_vl.py directly
    qwen = find_file("qwen2_5_vl.py")
    if qwen:
        print(f"[patch] Found: {qwen}")
        if patch_file(qwen):
            patched = True

    if not patched:
        print("[patch] ERROR: Could not find or patch F.scaled_dot_product_attention in any file.")
        sys.exit(1)

    print("[patch] Done.")


if __name__ == "__main__":
    main()
