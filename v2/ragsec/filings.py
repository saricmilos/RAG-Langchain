"""Read 10-K and 10-Q filings from the EDGAR full-submission files v1 downloaded.

Each submission holds the primary document (inline XBRL HTML) plus exhibits,
XBRL schemas and uuencoded images. v1 embedded the whole file; v2 keeps only the
primary document and turns it into ordered blocks: paragraphs and tables, each
with its Part/Item and the page number printed in the filing's footer.

The same HTML also carries the filing's tagged XBRL facts (ix:nonFraction),
which give the evaluation set its numeric answers.
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from lxml import html as lxml_html

from .config import COMPANIES, RAW_DIRS

UTF8_PARSER = lxml_html.HTMLParser(encoding="utf-8")

BLOCK_TAGS = {
    "p", "div", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "br", "center",
    "blockquote", "section", "article", "body", "font",
}
SKIP_TAGS = {"script", "style", "head", "title", "ix:header"}

ITEM_RE = re.compile(r"^item\s*(\d{1,2}[a-c]?)\s*[.:\-—–]?\s*(.*)$", re.I)
PART_RE = re.compile(r"^part\s+(i{1,3}|iv)\b\.?\s*(.*)$", re.I)
FOOTER_NUM_RE = re.compile(r"(?:^|[|\s])((?:[A-Z]{1,2}-)?\d{1,3})\s*$")
FOOTER_ONLY_RE = re.compile(r"^((?:[A-Z]{1,2}-)?\d{1,3})\.?$")   # Alphabet prints "20."


@dataclass
class Block:
    kind: str          # "text" or "table"
    text: str
    page: str | None   # page number as printed in the footer ("4", "K-19"), when there is one
    part: str          # "I", "II", ...
    item: str          # "1A", "7", ...
    item_title: str    # "Risk Factors"


@dataclass
class Filing:
    ticker: str
    company: str
    form: str            # "10-K" or "10-Q"
    accession: str
    cik: str
    period_end: date
    fiscal_year: int
    fiscal_period: str   # "FY", "Q1", "Q2", "Q3"
    primary_file: str
    blocks: list[Block] = field(default_factory=list)
    facts: list[dict] = field(default_factory=list)

    @property
    def doc_id(self) -> str:
        return f"{self.ticker}-{self.form}-{self.period_end.isoformat()}"

    @property
    def url(self) -> str:
        return (
            f"https://www.sec.gov/Archives/edgar/data/{int(self.cik)}/"
            f"{self.accession.replace('-', '')}/{self.primary_file}"
        )

    @property
    def period_label(self) -> str:
        end = f"{self.period_end:%B} {self.period_end.day}, {self.period_end.year}"
        if self.form == "10-K":
            return f"fiscal year {self.fiscal_year} (ended {end})"
        return f"fiscal {self.fiscal_year} {self.fiscal_period} (quarter ended {end})"


def find_submissions(forms=("10-K", "10-Q")) -> list[Path]:
    """Every full-submission.txt for the companies in COMPANIES, one per accession."""
    seen, paths = set(), []
    for root in RAW_DIRS:
        for ticker in COMPANIES:
            for form in forms:
                for p in sorted((root / ticker / form).glob("*/full-submission.txt")):
                    if p.parent.name not in seen:
                        seen.add(p.parent.name)
                        paths.append(p)
    return paths


def _header_value(header: str, key: str) -> str:
    m = re.search(rf"^\s*{re.escape(key)}:\s*(.+)$", header, re.M)
    return m.group(1).strip() if m else ""


def _primary_document(raw: str) -> tuple[str, str]:
    """(file name, HTML) of the first <DOCUMENT> in the submission."""
    start = raw.index("<DOCUMENT>")
    end = raw.index("</DOCUMENT>", start)
    doc = raw[start:end]
    name = re.search(r"<FILENAME>(.+)", doc).group(1).strip()
    body = doc[doc.index("<TEXT>") + 6:]
    return name, body.replace("</TEXT>", "")


def _clean(text: str) -> str:
    text = html.unescape(text).replace("\xa0", " ").replace("​", "")
    return re.sub(r"\s+", " ", text).strip()


def _is_hidden(el) -> bool:
    style = (el.get("style") or "").replace(" ", "").lower()
    return "display:none" in style


def _is_page_break(el) -> bool:
    style = (el.get("style") or "").replace(" ", "").lower()
    return "page-break-after:always" in style or "page-break-before:always" in style or "break-before:page" in style


def _table_rows(table) -> list[str]:
    """Table rows as 'cell | cell | cell', with '$', '%' and ')' glued to their numbers."""
    rows = []
    for tr in table.iter("tr"):
        cells = [_clean(td.text_content()) for td in tr if isinstance(td.tag, str) and td.tag in ("td", "th")]
        merged: list[str] = []
        for c in cells:
            if not c:
                continue
            if merged and c in (")", "%", ")%", "%)"):
                merged[-1] += c
            elif merged and merged[-1] in ("$", "(", "$(", "($"):
                merged[-1] += c
            else:
                merged.append(c)
        if merged and any(ch.isalnum() for ch in "".join(merged)):
            rows.append(" | ".join(merged))
    return rows


def _walk(el, out: list) -> None:
    tag = el.tag if isinstance(el.tag, str) else ""
    if tag in SKIP_TAGS or (tag and _is_hidden(el)):
        return
    if tag == "table":
        rows = _table_rows(el)
        if len(rows) == 1:
            # One-row tables are layout, e.g. Amazon's "Item 1A. | Risk Factors" headings.
            out.extend([("nl", None), ("text", rows[0].replace(" | ", " ")), ("nl", None)])
        elif rows:
            out.append(("table", rows))
        return
    if tag and _is_page_break(el) and "before" in (el.get("style") or "").lower():
        out.append(("break", None))
    is_block = tag in BLOCK_TAGS
    if is_block:
        out.append(("nl", None))
    if el.text and tag:
        out.append(("text", el.text))
    for child in el:
        _walk(child, out)
        if child.tail:
            out.append(("text", child.tail))
    if is_block:
        out.append(("nl", None))
    if tag and _is_page_break(el) and "after" in (el.get("style") or "").lower():
        out.append(("break", None))


def _blocks_from_html(root) -> list[tuple[str, str | list[str], int]]:
    """Raw (kind, content, page_index) blocks in document order."""
    events: list = []
    _walk(root, events)
    blocks, buf, page_idx = [], [], 0

    def flush():
        text = _clean("".join(buf))
        buf.clear()
        if text:
            blocks.append(("text", text, page_idx))

    for kind, val in events:
        if kind == "text":
            buf.append(val)
        elif kind == "nl":
            flush()
        elif kind == "table":
            flush()
            blocks.append(("table", val, page_idx))
        elif kind == "break":
            flush()
            page_idx += 1
    flush()
    return blocks


def _assign_pages_and_sections(raw_blocks) -> list[Block]:
    # The printed page number is the trailing number of the last short block on a page.
    by_page: dict[int, list[int]] = {}
    for i, (kind, _, p) in enumerate(raw_blocks):
        by_page.setdefault(p, []).append(i)
    printed: dict[int, str | None] = {}
    footer_ids: set[int] = set()
    for p, ids in by_page.items():
        printed[p] = None
        for i in reversed(ids[-3:]):
            kind, content, _ = raw_blocks[i]
            if kind == "text" and len(content) <= 80:
                m = FOOTER_ONLY_RE.match(content) or FOOTER_NUM_RE.search(content)
                if m:
                    printed[p] = m.group(1)
                    footer_ids.add(i)
                    break

    out: list[Block] = []
    part, item, title = "", "", "Cover and table of contents"
    for i, (kind, content, p) in enumerate(raw_blocks):
        if i in footer_ids:
            continue
        if kind == "text":
            if len(content) <= 160:
                pm = PART_RE.match(content)
                im = ITEM_RE.match(content)
                if pm and not re.search(r"\d{1,3}$", content):
                    part = pm.group(1).upper()
                elif im and not re.search(r"\s\d{1,3}$", content):
                    item = im.group(1).upper()
                    title = im.group(2).strip(" .") or title
            out.append(Block("text", content, printed.get(p), part, item, title))
        else:
            out.append(Block("table", "\n".join(content), printed.get(p), part, item, title))
    return _fill_item_titles(out)


def _fill_item_titles(blocks: list[Block]) -> list[Block]:
    """'Item 1A.' and 'Risk Factors' often sit in separate blocks: take the next block as the title."""
    for i, b in enumerate(blocks[:-1]):
        if b.kind == "text" and ITEM_RE.match(b.text) and not ITEM_RE.match(b.text).group(2):
            nxt = blocks[i + 1]
            if nxt.kind == "text" and len(nxt.text) < 120:
                for later in blocks[i:]:
                    if later.item != b.item:
                        break
                    later.item_title = nxt.text.strip(" .")
    return blocks


# ---------------------------------------------------------------- XBRL facts

def _parse_contexts(root) -> dict[str, dict]:
    contexts = {}
    for ctx in root.iter("xbrli:context"):
        cid = ctx.get("id")
        start = ctx.findtext(".//xbrli:startdate")
        end = ctx.findtext(".//xbrli:enddate")
        instant = ctx.findtext(".//xbrli:instant")
        has_segment = ctx.find(".//xbrli:segment") is not None
        contexts[cid] = {
            "start": start.strip() if start else None,
            "end": (end or instant or "").strip() or None,
            "instant": instant is not None,
            "segment": has_segment,
        }
    return contexts


def _fact_value(el) -> float | None:
    shown = _clean(el.text_content())
    fmt = (el.get("format") or "").lower()
    if not shown or shown in ("-", "—", "–") or "zerodash" in fmt or "fixed-zero" in fmt:
        return 0.0 if shown else None
    digits = shown.replace(",", "").replace(" ", "")
    if "comma-decimal" in fmt or "numcommadecimal" in fmt:
        digits = shown.replace(".", "").replace(" ", "").replace(",", ".")
    try:
        value = float(re.sub(r"[^0-9.]", "", digits))
    except ValueError:
        return None
    value *= 10 ** int(el.get("scale") or 0)
    if el.get("sign") == "-":
        value = -value
    return value


def _parse_facts(root) -> tuple[list[dict], dict[str, str]]:
    contexts = _parse_contexts(root)
    facts = []
    for el in root.iter("ix:nonfraction"):
        ctx = contexts.get(el.get("contextref"), {})
        value = _fact_value(el)
        if value is None:
            continue
        facts.append({
            "concept": el.get("name"),
            "value": value,
            "shown": _clean(el.text_content()),
            "scale": int(el.get("scale") or 0),
            "unit": (el.get("unitref") or "").lower(),
            "start": ctx.get("start"),
            "end": ctx.get("end"),
            "instant": ctx.get("instant", False),
            "segment": ctx.get("segment", True),
        })
    dei = {}
    for el in root.iter("ix:nonnumeric"):
        name = el.get("name") or ""
        if name.startswith("dei:"):
            dei.setdefault(name[4:], _clean(el.text_content()))
    return facts, dei


# ---------------------------------------------------------------- entry point

def load_filing(path: Path) -> Filing:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    header = raw[: raw.index("</SEC-HEADER>")]
    primary_file, doc_html = _primary_document(raw)
    ticker = path.parents[2].name
    form = path.parents[1].name
    period = _header_value(header, "CONFORMED PERIOD OF REPORT")
    period_end = date(int(period[:4]), int(period[4:6]), int(period[6:8]))

    root = lxml_html.fromstring(doc_html.encode("utf-8"), parser=UTF8_PARSER)
    facts, dei = _parse_facts(root)
    fiscal_year = int(re.sub(r"\D", "", dei.get("DocumentFiscalYearFocus", "")) or period_end.year)
    fiscal_period = (dei.get("DocumentFiscalPeriodFocus") or ("FY" if form == "10-K" else "")).upper()

    filing = Filing(
        ticker=ticker,
        company=COMPANIES[ticker]["name"],
        form=form,
        accession=path.parent.name,
        cik=_header_value(header, "CENTRAL INDEX KEY"),
        period_end=period_end,
        fiscal_year=fiscal_year,
        fiscal_period=fiscal_period,
        primary_file=primary_file,
        facts=facts,
    )
    filing.blocks = _assign_pages_and_sections(_blocks_from_html(root))
    return filing
