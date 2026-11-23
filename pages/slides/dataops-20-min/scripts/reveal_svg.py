#!/usr/bin/env python3
"""Split a Graphviz SVG into cumulative reveal stages.

The input is the SVG exported by Graphviz (for example the graph produced by
``skrub``'s ``.skb.draw()``).  Each ``--stage`` lists the nodes that should be
revealed at that step; stages accumulate, so stage 3 contains stages 1 and 2.
A node can be referenced by its Graphviz id (``node_0``) or by the label shown
in the graph (``Var 'data_path'``).  Edges are added automatically as soon as
both of their endpoints are visible.

Typical use:

    python scripts/reveal_svg.py images/base_ex.svg \
        --out-dir images --prefix base_ex \
        --stage "Var 'data_path'" \
        --stage "Call 'read_csv'" \
        --stage "X: CallMethod 'drop'" "y: GetItem 'target'" \
        --stage "Apply StandardScaler" "Apply LogisticRegression"

This writes ``images/base_ex_1.svg`` ... ``images/base_ex_4.svg``.

Use ``--list-nodes`` to print the available node ids and labels.
"""

from __future__ import annotations

import argparse
import copy
import html
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"

ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def normalise(text: str) -> str:
    return " ".join(text.split())


def node_label(node: ET.Element) -> str:
    labels = [
        "".join(el.itertext())
        for el in node.iter()
        if local_name(el.tag) == "text"
    ]
    return normalise(" ".join(labels))


def classes(element: ET.Element) -> set[str]:
    return set((element.get("class") or "").split())


def parse_edge_endpoints(raw_svg: str) -> dict[str, tuple[str, str]]:
    """Recover edge endpoints from Graphviz's ``<!-- a-&gt;b -->`` comments."""
    comment = re.compile(
        r"<!--\s*(.*?)\s*-->\s*<g\b[^>]*?\bid=\"([^\"]+)\"", re.DOTALL
    )
    endpoints: dict[str, tuple[str, str]] = {}
    for text, group_id in comment.findall(raw_svg):
        text = normalise(html.unescape(text))
        for separator in ("->", "--"):
            if separator in text:
                source, target = text.split(separator, 1)
                endpoints[group_id] = (source.strip(), target.strip())
                break
    return endpoints


def unwrap_anchors(parent: ET.Element) -> None:
    """Replace ``<a>`` wrappers by their children so links become inert."""
    for child in list(parent):
        if local_name(child.tag) == "a":
            position = list(parent).index(child)
            parent.remove(child)
            for offset, grandchild in enumerate(list(child)):
                parent.insert(position + offset, grandchild)
        else:
            unwrap_anchors(child)


def describe(node: ET.Element) -> str:
    return f"  {node.get('id', '?')}: {node_label(node)}"


def resolve_node(
    token: str,
    nodes: list[ET.Element],
    by_id: dict[str, ET.Element],
    by_label: dict[str, list[ET.Element]],
) -> ET.Element:
    if token in by_id:
        return by_id[token]

    key = normalise(token).casefold()
    matches = by_label.get(key, [])
    if len(matches) == 1:
        return matches[0]

    if not matches:
        matches = [node for node in nodes if key in node_label(node).casefold()]

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        rendered = "\n".join(describe(node) for node in matches)
        raise SystemExit(f"{token!r} is ambiguous, it matches:\n{rendered}")

    available = "\n".join(describe(node) for node in nodes)
    raise SystemExit(f"No node matches {token!r}. Available nodes:\n{available}")


def visible_stage(
    root: ET.Element,
    graph: ET.Element,
    revealed: set[str],
    endpoints: dict[str, tuple[str, str]],
    keep_background: bool,
    keep_links: bool,
    missing_edges: set[str],
) -> ET.Element:
    new_root = ET.Element(root.tag, dict(root.attrib))
    new_graph = ET.SubElement(new_root, graph.tag, dict(graph.attrib))

    for child in graph:
        kind = classes(child)
        if "node" in kind:
            if child.get("id") in revealed:
                new_graph.append(copy.deepcopy(child))
        elif "edge" in kind:
            edge_id = child.get("id", "")
            edge = endpoints.get(edge_id)
            if edge is None:
                missing_edges.add(edge_id)
            elif edge[0] in revealed and edge[1] in revealed:
                new_graph.append(copy.deepcopy(child))
        elif keep_background:
            new_graph.append(copy.deepcopy(child))

    if not keep_links:
        unwrap_anchors(new_graph)

    try:
        ET.indent(new_root, space="  ")
    except AttributeError:
        pass
    return new_root


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="Graphviz SVG file")
    parser.add_argument(
        "--stage",
        action="append",
        nargs="+",
        metavar="NODE",
        help="nodes to reveal at this step (repeat the option for each step)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("."),
        help="directory for the generated SVGs (default: current directory)",
    )
    parser.add_argument(
        "--prefix",
        help="output filename prefix (default: input filename stem)",
    )
    parser.add_argument(
        "--keep-background",
        action="store_true",
        help="keep the Graphviz background polygon in every stage",
    )
    parser.add_argument(
        "--keep-links",
        action="store_true",
        help="keep the <a> tooltip wrappers around the nodes",
    )
    parser.add_argument(
        "--list-nodes",
        action="store_true",
        help="list the nodes found in the SVG and exit",
    )
    args = parser.parse_args()

    raw_svg = args.input.read_text(encoding="utf-8")
    root = ET.fromstring(raw_svg)

    graph = next(
        (el for el in root if local_name(el.tag) == "g"),
        next((el for el in root.iter() if local_name(el.tag) == "g"), None),
    )
    if graph is None:
        raise SystemExit(f"{args.input}: could not find the Graphviz graph")

    nodes = [el for el in graph if "node" in classes(el)]
    if not nodes:
        raise SystemExit(f"{args.input}: no graph nodes found")

    by_id = {node.get("id"): node for node in nodes if node.get("id")}
    by_label: dict[str, list[ET.Element]] = {}
    for node in nodes:
        by_label.setdefault(node_label(node).casefold(), []).append(node)

    if args.list_nodes:
        for node in nodes:
            print(describe(node))
        return

    if not args.stage:
        parser.error("at least one --stage is required")

    endpoints = parse_edge_endpoints(raw_svg)
    prefix = args.prefix or args.input.stem
    args.out_dir.mkdir(parents=True, exist_ok=True)

    revealed: set[str] = set()
    missing_edges: set[str] = set()
    for index, tokens in enumerate(args.stage, start=1):
        for token in tokens:
            node = resolve_node(token, nodes, by_id, by_label)
            revealed.add(node.get("id"))

        stage_root = visible_stage(
            root,
            graph,
            revealed,
            endpoints,
            args.keep_background,
            args.keep_links,
            missing_edges,
        )
        output = args.out_dir / f"{prefix}_{index}.svg"
        ET.ElementTree(stage_root).write(
            output, encoding="UTF-8", xml_declaration=True
        )
        print(f"wrote {output}")

    if missing_edges:
        names = ", ".join(sorted(missing_edges))
        print(
            f"warning: no endpoints found for {names}; those edges were skipped",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
