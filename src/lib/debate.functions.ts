import { createServerFn } from "@tanstack/react-start";
import { z } from "zod";
import { callGroq } from "./groq-client";

export type DebateResult = {
  mode: "debate";
  mainViewpoint: string[];
  counterargument: string[];
  balancedConclusion: string;
};

function extractJson(raw: string): string {
  let s = raw.trim();
  s = s.replace(/^```(?:json)?\s*/i, "").replace(/```\s*$/i, "").trim();
  const f = s.search(/[{[]/);
  if (f === -1) throw new Error("no json");
  const open = s[f];
  const close = open === "{" ? "}" : "]";
  const l = s.lastIndexOf(close);
  if (l <= f) throw new Error("no json");
  return s.slice(f, l + 1);
}

const contextSchema = z.object({
  title: z.string(),
  category: z.string(),
  executiveSummary: z.string(),
  keyInsights: z.array(z.object({ title: z.string(), body: z.string() })),
});

const debateSchema = z.object({
  mainViewpoint: z.array(z.string()).min(2).max(4),
  counterargument: z.array(z.string()).min(2).max(4),
  // The model occasionally returns this as an array of sentences instead of
  // one joined string, even though the prompt asks for a string. Accept
  // either shape and normalize to a string rather than failing validation.
  balancedConclusion: z
    .union([z.string(), z.array(z.string())])
    .transform((v) => (Array.isArray(v) ? v.join(" ") : v)),
});

export const generateDebate = createServerFn({ method: "POST" })
  .inputValidator((i: unknown) => z.object({ context: contextSchema }).parse(i))
  .handler(async ({ data }): Promise<DebateResult> => {
    const ctx = `Video: ${data.context.title}
Category: ${data.context.category}
Summary:
${data.context.executiveSummary}
Insights:
${data.context.keyInsights.map((k) => `- ${k.title}: ${k.body}`).join("\n")}`;

    // Applied to EVERY video, including tutorials/factual content. For
    // opinion-based videos, counterargument is a genuine opposing view. For
    // instructional/factual videos, counterargument is framed as a real
    // alternative method, a common misconception, or a genuine limitation
    // of the approach shown — never an invented or strawman disagreement.
    const debateSystem = `You are an educational critical-thinking assistant. Based on the video below, produce a balanced exploration of perspectives. Rules: mainViewpoint = 2-4 short bullet points summarizing the video's primary argument or approach. counterargument = 2-4 short bullet points giving a realistic, thoughtful, educational opposing perspective — not a strawman. If the video is opinion-based, this is a genuine opposing viewpoint. If the video is instructional or factual (e.g. a tutorial), frame this as a genuinely different valid method, an important limitation, or a common misconception about the approach shown — still a real, substantive perspective, never invented just to fill the section. balancedConclusion = 2-3 neutral sentences comparing both views and when each may be valid. Return JSON only in this shape: {"mainViewpoint":[string,...],"counterargument":[string,...],"balancedConclusion":string}`;

    const content = await callGroq({
           models: ["openai/gpt-oss-20b", "openai/gpt-oss-120b"],
      keyOffset: 1,
      messages: [
        { role: "system", content: debateSystem },
        { role: "user", content: ctx },
      ],
      maxTokens: 700,
      temperature: 0.6,
    });

    let parsed: unknown;
    try {
      parsed = JSON.parse(extractJson(content));
    } catch {
      throw new Error("AI returned malformed output.");
    }

    const r = debateSchema.parse(parsed);
    return { mode: "debate", ...r };
  });
