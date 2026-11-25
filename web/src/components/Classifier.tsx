import { useEffect, useState } from "react";
import type { Classification } from "../model";

const NAMES: Record<string, string> = {
  alligator: "Alligator",
  caiman: "Caiman",
  crocodile: "Crocodile",
  gharial: "Gharial",
};

type State =
  | { step: "idle" }
  | { step: "working"; photo: string }
  | { step: "done"; photo: string; result: Classification }
  | { step: "error"; photo?: string; message: string };

export function Classifier() {
  const [state, setState] = useState<State>({ step: "idle" });

  const photo = state.step === "idle" ? undefined : state.photo;
  useEffect(() => () => {
    if (photo) URL.revokeObjectURL(photo);
  }, [photo]);

  async function onFile(file: File | undefined) {
    if (!file) return;
    const url = URL.createObjectURL(file);
    setState({ step: "working", photo: url });
    try {
      // Loaded on first upload so the page itself stays small.
      const { classify } = await import("../model");
      setState({ step: "done", photo: url, result: await classify(file) });
    } catch (err) {
      setState({ step: "error", photo: url, message: err instanceof Error ? err.message : String(err) });
    }
  }

  return (
    <section className="classifier">
      <p className="intro">
        Upload a photo of an alligator, caiman, crocodile or gharial and the model will tell you which it is.
      </p>
      <label className="upload">
        <input type="file" accept="image/*" onChange={(e) => onFile(e.target.files?.[0])} />
        {state.step === "idle" ? "Choose a photo" : "Choose another photo"}
      </label>

      {photo && <img className="photo" src={photo} alt="Your upload" />}

      {state.step === "working" && <p className="status">Looking closely…</p>}
      {state.step === "error" && <p className="status error">Something went wrong: {state.message}</p>}
      {state.step === "done" && <Result result={state.result} />}
    </section>
  );
}

function Result({ result }: { result: Classification }) {
  const { verdict, confidences } = result;
  return (
    <div className="result">
      {verdict.kind === "group" ? (
        <h2>
          It's a {NAMES[verdict.group].toLowerCase()}{" "}
          <span className="muted">({Math.round(verdict.confidence * 100)}% sure)</span>
        </h2>
      ) : (
        <h2>
          Can't tell
          <span className="muted"> — this may not be a crocodilian, or the photo is unclear.</span>
        </h2>
      )}
      <ul className="bars">
        {confidences.map(({ group, confidence }) => (
          <li key={group}>
            <span className="label">{NAMES[group]}</span>
            <span className="bar">
              <span style={{ width: `${(confidence * 100).toFixed(1)}%` }} />
            </span>
            <span className="pct">{Math.round(confidence * 100)}%</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
