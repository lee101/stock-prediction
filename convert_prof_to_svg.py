#!/usr/bin/env python3
"""Convert .prof to SVG flamegraph manually."""
import pstats
import sys
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom

def generate_flamegraph(profile_file, output_svg):
    """Generate a flamegraph manually from profile stats."""
    stats = pstats.Stats(profile_file)
    stats.strip_dirs()

    total_time = sum(timing[2] for timing in stats.stats.values())

    svg = Element('svg', {
        'version': '1.1',
        'width': '1600',
        'height': '800',
        'xmlns': 'http://www.w3.org/2000/svg'
    })

    defs = SubElement(svg, 'defs')
    style = SubElement(defs, 'style', {'type': 'text/css'})
    style.text = '.func_g:hover { stroke:black; stroke-width:0.5; cursor:pointer; }'

    sorted_stats = sorted(stats.stats.items(), key=lambda x: x[1][3], reverse=True)

    y_offset = 750
    x_offset = 10
    height = 18
    scale = 1580 / total_time if total_time > 0 else 1

    for i, (func_key, (cc, nc, tt, ct, callers)) in enumerate(sorted_stats[:40]):
        filename, line, func_name = func_key

        width = ct * scale
        if width < 1:
            continue

        g = SubElement(svg, 'g', {'class': 'func_g'})

        title = SubElement(g, 'title')
        percentage = (ct / total_time * 100) if total_time > 0 else 0
        samples = int(ct * 1000)
        title.text = f"{func_name}\n{filename}:{line}\n{samples} ms ({percentage:.2f}%)\ncalls: {cc}"

        colors = ['rgb(250,128,114)', 'rgb(250,200,100)', 'rgb(100,200,250)',
                  'rgb(150,250,150)', 'rgb(250,150,250)', 'rgb(200,100,200)',
                  'rgb(255,180,100)', 'rgb(180,255,180)', 'rgb(180,180,255)']
        color = colors[i % len(colors)]

        rect = SubElement(g, 'rect', {
            'x': str(x_offset),
            'y': str(y_offset - i * 20),
            'width': str(width),
            'height': str(height),
            'fill': color,
            'stroke': 'white',
            'stroke-width': '0.5'
        })

        text = SubElement(g, 'text', {
            'x': str(x_offset + 5),
            'y': str(y_offset - i * 20 + 13),
            'font-size': '11',
            'font-family': 'monospace',
            'fill': 'black'
        })
        display_name = f"{func_name} ({percentage:.1f}%)"
        text.text = display_name[:100]

    xml_str = minidom.parseString(tostring(svg)).toprettyxml(indent='  ')

    with open(output_svg, 'w') as f:
        f.write(xml_str)

    print(f"Flamegraph saved to {output_svg}")

if __name__ == '__main__':
    profile_file = sys.argv[1] if len(sys.argv) > 1 else 'optimization_test.prof'
    output_svg = sys.argv[2] if len(sys.argv) > 2 else profile_file.replace('.prof', '.svg')
    generate_flamegraph(profile_file, output_svg)
