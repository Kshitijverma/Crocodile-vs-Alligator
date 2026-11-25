import { useEffect, useState } from "react";

interface Credit {
  group: string;
  attribution: string;
  license: string;
  url: string;
}

export function Credits() {
  const [credits, setCredits] = useState<Credit[] | null>(null);

  useEffect(() => {
    fetch("/credits.json")
      .then((r) => r.json())
      .then(setCredits)
      .catch(() => setCredits([]));
  }, []);

  return (
    <section className="credits">
      <h2>Credits</h2>
      <p>
        The model was trained on research-grade observations from{" "}
        <a href="https://www.inaturalist.org">iNaturalist</a>, used under their CC0, CC BY and CC BY-NC
        licences. Thank you to every observer below.
      </p>
      {credits === null ? (
        <p className="status">Loading…</p>
      ) : (
        <ul>
          {credits.map((c) => (
            <li key={c.url}>
              <a href={c.url}>{c.attribution || "Unknown observer"}</a>{" "}
              <span className="muted">
                {c.group} · {c.license.toUpperCase()}
              </span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
