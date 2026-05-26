"""tools/pattern_tool.py — View behavioral patterns."""
from core.store import store


class ViewPatternTool:
    name        = "view_patterns"
    description = "View active behavioral patterns for a headmate."

    async def execute(self, args, session_id, headmate, llm) -> str:
        about = args.get("headmate") or headmate or ""
        if not about:
            return "no headmate specified"

        patterns = store.get_patterns(about, min_confidence=0.2)
        if not patterns:
            return f"no patterns on file for {about}"

        lines = []
        for p in patterns:
            lines.append(
                f"[{p.get('action','?').upper()}] {p.get('pattern_type','?')} "
                f"conf={p.get('confidence',0):.2f} "
                f"pts={p.get('data_points',0)} "
                f"quality={p.get('outcome_quality_avg',0):.2f}"
            )
            if p.get("approach"):
                lines.append(f"  → {p['approach'][:80]}")
        return "\n".join(lines)
