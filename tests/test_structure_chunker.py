from rag_engine.documents.structure_chunker import StructureChunker


def test_markdown_sections_produce_parents_and_children():
    text = """# Introduction

Photosynthesis converts light into chemical energy. It happens in chloroplasts.

# Methods

We measured oxygen production under different light intensities.

## Results

Oxygen increased with light intensity until saturation.
"""
    nodes = StructureChunker(child_chunk_size=120, child_overlap=10, parent_chunk_size=400).split(
        text
    )

    parents = [node for node in nodes if node.role == "parent"]
    children = [node for node in nodes if node.role == "child"]

    assert parents
    assert children
    assert any(node.section_title == "Introduction" for node in parents)
    assert any(node.section_path == ["Methods", "Results"] for node in parents)
    # Every child points at a parent node index.
    assert all(node.parent_key is not None for node in children)
    assert all(0 <= node.parent_key < len(nodes) for node in children)
    assert all(nodes[node.parent_key].role == "parent" for node in children)


def test_page_markers_alone_are_not_substantive():
    nodes = StructureChunker().split("[Page 1]\n[Page 2]\n")
    assert nodes == []


def test_page_marker_before_a_heading_is_carried_into_the_section():
    nodes = StructureChunker().split("[Page 3]\n# Introduction\n\nThe actual text.")
    assert nodes
    assert all("[Page 3]" in node.content for node in nodes)
    assert nodes[0].section_title == "Introduction"
