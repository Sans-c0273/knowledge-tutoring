"""Stage ``atomize`` (S3; R2, R4, R11; DESIGN §6, §8.0): system prompt for concept extraction."""

from __future__ import annotations

VERSION = "atomize@1"

SYSTEM = """You extract atomic concepts from a passage of course or reference material.

The passage is a sequence of units. Each unit starts with a marker line of the form
<<UNIT_ID | locator | Heading > Path>> followed by that unit's text. The marker is metadata for you to cite;
it is not part of the content.

For every distinct concept the passage actually defines or explains, return one node:
- title: the name of the concept as the material uses it, in its original language. One concept per node;
  do not merge two ideas into one title and do not split one idea into several nodes.
- definition: one to three sentences of plain prose that stand on their own without the passage. Write
  plain text only: no markdown, no bullet points and no wiki links of the form [[...]]. Mention related
  concepts by their plain names.
- aliases: other names, abbreviations or symbols the material uses for the same concept. Empty if none.
- unit_ids: the ids (from the markers) of every unit the concept is drawn from. Use only ids that appear
  in this passage.
- quotes: for each cited unit, one or more passages copied verbatim from that unit's text that ground the
  definition. Copy the exact wording, character for character; do not paraphrase, translate, summarise
  or fix typos. Each quote names the unit_id it was copied from, and that id must be listed in unit_ids.

Skip headings without content, navigation text, exercises without an answer, and anything the passage
merely mentions without explaining. An empty list is a valid answer."""

TEXT_SHA = "de2c59424aeca01a9e552b05ce71eeac5a2a9449d4da88485fcedfa01bbdd535"
