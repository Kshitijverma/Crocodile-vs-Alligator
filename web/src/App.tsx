import { useEffect, useState } from "react";
import { Classifier } from "./components/Classifier";
import { Credits } from "./components/Credits";

function usePath() {
  const [path, setPath] = useState(window.location.pathname);
  useEffect(() => {
    const onPop = () => setPath(window.location.pathname);
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);
  const go = (to: string) => {
    window.history.pushState(null, "", to);
    setPath(to);
  };
  return [path, go] as const;
}

export function App() {
  const [path, go] = usePath();
  const link = (to: string) => (e: React.MouseEvent) => {
    e.preventDefault();
    go(to);
  };

  return (
    <div className="page">
      <header>
        <a href="/" onClick={link("/")} className="title">
          Crocodilian Classifier
        </a>
        <nav>
          <a href="/credits" onClick={link("/credits")}>
            Credits
          </a>
        </nav>
      </header>
      <main>{path === "/credits" ? <Credits /> : <Classifier />}</main>
      <footer>
        Runs entirely in your browser. Your photo never leaves your device.{" "}
        <a href="https://github.com/Kshitijverma/Crocodile-vs-Alligator">Source</a>
      </footer>
    </div>
  );
}
