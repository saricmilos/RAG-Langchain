"""Cut parsed filings into chunks that respect Items, tables and the embedder's window.

v1 cut raw SGML every 2,000 characters, so a chunk could start inside an XBRL tag
and end in the middle of a table, and the embedder read only its first 256 tokens.
Here a chunk never crosses an Item, a table is kept whole (or split by rows with
its header rows repeated), and every chunk carries a one-line header naming the
company, form, period and Item, which is embedded and indexed with it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict

from .filings import Block, Filing

MAX_TEXT_CHARS = 1400
MAX_TABLE_CHARS = 900
CAPTION_MAX_CHARS = 300

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(“\"])")


@dataclass
class Chunk:
    id: str
    doc_id: str
    ticker: str
    company: str
    form: str
    fiscal_year: int
    fiscal_period: str
    period_end: str
    period_label: str
    url: str
    part: str
    item: str
    item_title: str
    page: str | None
    kind: str
    text: str

    @property
    def header(self) -> str:
        where = f"Item {self.item}. {self.item_title}" if self.item else self.item_title
        if self.part:
            where = f"Part {self.part}, {where}"
        return f"{self.company} ({self.ticker}) {self.form} for {self.period_label}. {where}."

    @property
    def indexed_text(self) -> str:
        """What the embedder and BM25 see: the header line, then the body."""
        return f"{self.header}\n{self.text}"

    def to_dict(self) -> dict:
        return asdict(self)


def _split_long_text(text: str, limit: int) -> list[str]:
    parts, buf = [], ""
    for sentence in SENTENCE_RE.split(text):
        while len(sentence) > limit:            # a single run-on "sentence" (lists, legal text)
            cut = sentence.rfind(" ", 0, limit)
            cut = cut if cut > limit // 2 else limit
            parts.append((buf + " " + sentence[:cut]).strip() if buf else sentence[:cut])
            buf, sentence = "", sentence[cut:].strip()
        if buf and len(buf) + 1 + len(sentence) > limit:
            parts.append(buf)
            buf = sentence
        else:
            buf = f"{buf} {sentence}".strip()
    if buf:
        parts.append(buf)
    return parts


def _split_table(rows: list[str], caption: str, limit: int) -> list[str]:
    # A row longer than the limit (audit-matter tables hold whole paragraphs in one cell) is split like text.
    rows = [piece for row in rows for piece in (_split_long_text(row, limit) if len(row) > limit else [row])]
    head = [r for r in rows[: min(2, len(rows))] if len(r) <= limit // 3]
    body = rows[len(head):]
    prefix = ([caption] if caption else []) + head
    pieces, cur = [], list(prefix)
    for row in body:
        if len("\n".join(cur + [row])) > limit and len(cur) > len(prefix):
            pieces.append("\n".join(cur))
            cur = list(prefix)
        cur.append(row)
    if len(cur) > len(prefix) or not pieces:
        pieces.append("\n".join(cur))
    return pieces


def chunk_filing(filing: Filing) -> list[Chunk]:
    chunks: list[Chunk] = []

    def emit(kind: str, text: str, block: Block) -> None:
        chunks.append(Chunk(
            id=f"{filing.doc_id}#{len(chunks):04d}",
            doc_id=filing.doc_id,
            ticker=filing.ticker,
            company=filing.company,
            form=filing.form,
            fiscal_year=filing.fiscal_year,
            fiscal_period=filing.fiscal_period,
            period_end=filing.period_end.isoformat(),
            period_label=filing.period_label,
            url=filing.url,
            part=block.part,
            item=block.item,
            item_title=block.item_title,
            page=block.page,
            kind=kind,
            text=text,
        ))

    buf: list[str] = []
    buf_first: Block | None = None
    section = None

    def flush() -> None:
        nonlocal buf, buf_first
        if buf and buf_first is not None:
            emit("text", "\n".join(buf), buf_first)
        buf, buf_first = [], None

    blocks = filing.blocks
    for i, block in enumerate(blocks):
        key = (block.part, block.item)
        if key != section:
            flush()
            section = key

        if block.kind == "table":
            # A short line right before the table ("The following table shows ...") is its caption.
            caption = ""
            if buf and len(buf[-1]) <= CAPTION_MAX_CHARS:
                caption = buf.pop()
            flush()
            for piece in _split_table(block.text.split("\n"), caption, MAX_TABLE_CHARS):
                emit("table", piece, block)
            continue

        pieces = _split_long_text(block.text, MAX_TEXT_CHARS) if len(block.text) > MAX_TEXT_CHARS else [block.text]
        for piece in pieces:
            if buf and len("\n".join(buf)) + 1 + len(piece) > MAX_TEXT_CHARS:
                flush()
            if buf_first is None:
                buf_first = block
            buf.append(piece)
    flush()
    return chunks
