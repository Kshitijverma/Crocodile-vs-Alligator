import { describe, expect, it } from "vitest";
import { verdict } from "../src/verdict";

const GROUPS = ["alligator", "caiman", "crocodile", "gharial"];

describe("verdict", () => {
  it("names the Group whose Confidence reaches the Threshold", () => {
    expect(verdict([0.05, 0.05, 0.85, 0.05], GROUPS, 0.6)).toEqual({
      kind: "group",
      group: "crocodile",
      confidence: 0.85,
    });
  });

  it("says Can't tell when no Confidence reaches the Threshold", () => {
    expect(verdict([0.3, 0.25, 0.25, 0.2], GROUPS, 0.6)).toEqual({ kind: "cant-tell" });
  });

  it("names the Group when its Confidence equals the Threshold exactly", () => {
    expect(verdict([0.6, 0.2, 0.1, 0.1], GROUPS, 0.6)).toEqual({
      kind: "group",
      group: "alligator",
      confidence: 0.6,
    });
  });
});
