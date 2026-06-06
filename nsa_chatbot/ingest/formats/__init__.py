"""Format-specific knowledge lives here.

One module per input format / output format:

* :mod:`nsa_chatbot.ingest.formats.legal_text` — the markers and regexes that
  describe how *legal text* (sections, subsections) is structured. Used by
  the chunker and emitted by the eCFR fetcher so both ends agree.
* :mod:`nsa_chatbot.ingest.formats.ecfr` — eCFR XML schema knowledge (DIV8/HEAD/P/FP).
* :mod:`nsa_chatbot.ingest.formats.html_pages` — site-specific HTML extraction
  selectors and chrome-stripping rules.
* :mod:`nsa_chatbot.ingest.formats.pdf` — pypdf-based text extraction.
"""
