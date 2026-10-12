"""Step 3: embed every chunk with each embedder in the ablation (cached in work/emb).

    python scripts/03_embed.py oai-small bge-small minilm minilm-nohdr
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragsec.embed import EMBEDDERS, chunk_matrix, load_chunks


def main(keys: list[str]) -> None:
    chunks = load_chunks()
    for key in keys or list(EMBEDDERS):
        t = time.time()
        m = chunk_matrix(key, chunks)
        print(f"{key}: {m.shape} in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
