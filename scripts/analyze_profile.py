#!/usr/bin/env python3
"""Analyze Python profile files and output hot paths to markdown."""

import pstats
import sys
from pathlib import Path
from typing import List, Dict


class ProfileAnalyzer:
    """Parse .prof files and extract hot paths."""

    def __init__(self, profile_path: Path):
        self.profile_path = profile_path
        self.profile_data = []
        self.stats = None

    def parse(self) -> List[Dict]:
        """Parse profile file and extract performance data."""
        self.stats = pstats.Stats(str(self.profile_path))
        self.stats.strip_dirs()

        # Get the stats in a parseable format
        stats_dict = {}
        self.stats.calc_callees()

        for func, (cc, nc, tt, ct, callers) in self.stats.stats.items():
            filename, line, func_name = func
            stats_dict[func] = {
                'function': f"{func_name}",
                'filename': filename,
                'line': line,
                'func_name': func_name,
                'ncalls': nc,
                'tottime': tt,
                'percall': tt/nc if nc > 0 else 0,
                'cumtime': ct,
                'percall_cum': ct/nc if nc > 0 else 0,
                'location': f"{filename}:{line}"
            }

        # Convert to list and sort by cumulative time
        self.profile_data = list(stats_dict.values())
        self.profile_data.sort(key=lambda x: x['cumtime'], reverse=True)

        # Calculate percentages
        total_time = sum(x['cumtime'] for x in self.profile_data)
        for item in self.profile_data:
            item['percentage'] = (item['cumtime'] / total_time * 100) if total_time > 0 else 0

        return self.profile_data

    def format_markdown(self, top_n=50, hotspot_threshold=1.0) -> str:
        """Format profile data as markdown."""
        if not self.profile_data:
            self.parse()

        md_lines = [
            f"# Profile Analysis: {self.profile_path.name}",
            "",
            "## Summary",
            "",
            f"Total functions profiled: {len(self.profile_data)}",
            ""
        ]

        # Calculate total time
        total_time = sum(x.get('cumtime', 0) for x in self.profile_data)
        md_lines.append(f"Total execution time: {total_time:.3f} seconds")
        md_lines.append("")

        md_lines.extend([
            f"## Top {top_n} Time-Consuming Functions (Hot Paths)",
            "",
            "| Rank | Function | Location | Cumulative | Own Time | Calls | % Total |",
            "|------|----------|----------|------------|----------|-------|---------|"
        ])

        # Show top N functions
        for idx, item in enumerate(self.profile_data[:top_n], 1):
            func_name = item.get('func_name', 'Unknown')
            location = item.get('location', 'Unknown')

            # Truncate long names for table
            if len(func_name) > 40:
                func_name = func_name[:37] + "..."
            if len(location) > 50:
                location = "..." + location[-47:]

            cumtime = item.get('cumtime', 0)
            tottime = item.get('tottime', 0)
            ncalls = item.get('ncalls', 0)
            percentage = item.get('percentage', 0)

            md_lines.append(
                f"| {idx} | `{func_name}` | {location} | {cumtime:.3f}s | {tottime:.3f}s | {ncalls:,} | {percentage:.1f}% |"
            )

        # Add hotspots section (functions taking >threshold% of time)
        md_lines.extend([
            "",
            f"## Performance Hotspots (>{hotspot_threshold}% of total time)",
            "",
        ])

        hotspots = [f for f in self.profile_data if f.get('percentage', 0) >= hotspot_threshold]

        for item in hotspots[:30]:  # Limit to top 30 hotspots
            func_name = item.get('func_name', 'Unknown')
            location = item.get('location', 'Unknown')
            percentage = item.get('percentage', 0)
            cumtime = item.get('cumtime', 0)
            tottime = item.get('tottime', 0)
            ncalls = item.get('ncalls', 0)

            # Create visual bar
            bar_length = min(int(percentage), 50)
            bar = "█" * bar_length

            md_lines.extend([
                f"### `{func_name}` ({location})",
                f"{bar} **{percentage:.2f}%**",
                f"- **Cumulative time**: {cumtime:.3f}s",
                f"- **Own time**: {tottime:.3f}s",
                f"- **Calls**: {ncalls:,}",
                f"- **Time per call**: {item.get('percall_cum', 0):.6f}s",
                ""
            ])

        # Add recommendations section
        md_lines.extend([
            "",
            "## Optimization Recommendations",
            "",
            "Based on the profile data, consider optimizing:",
            ""
        ])

        # Top 5 by cumulative time
        for idx, item in enumerate(self.profile_data[:5], 1):
            func_name = item.get('func_name', 'Unknown')
            location = item.get('location', 'Unknown')
            percentage = item.get('percentage', 0)

            md_lines.append(
                f"{idx}. **{func_name}** ({location}) - {percentage:.1f}% of total time"
            )

        return "\n".join(md_lines)


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_profile.py <profile.prof> [output.md]")
        sys.exit(1)

    profile_path = Path(sys.argv[1])
    if not profile_path.exists():
        print(f"Error: Profile file not found: {profile_path}")
        sys.exit(1)

    analyzer = ProfileAnalyzer(profile_path)
    markdown = analyzer.format_markdown()

    if len(sys.argv) > 2:
        output_path = Path(sys.argv[2])
        output_path.write_text(markdown)
        print(f"Analysis written to: {output_path}")
    else:
        print(markdown)


if __name__ == '__main__':
    main()
