/** What the site tells the visitor about one photo (see CONTEXT.md). */
export type Verdict =
  | { kind: "group"; group: string; confidence: number }
  | { kind: "cant-tell" };

/** Turn per-Group Confidences into a Verdict. */
export function verdict(confidences: ArrayLike<number>, groups: string[], threshold: number): Verdict {
  let best = 0;
  for (let i = 1; i < confidences.length; i++) {
    if (confidences[i] > confidences[best]) best = i;
  }
  if (confidences[best] < threshold) return { kind: "cant-tell" };
  return { kind: "group", group: groups[best], confidence: confidences[best] };
}
