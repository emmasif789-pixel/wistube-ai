import { createServerFn } from "@tanstack/react-start";
import { z } from "zod";
import { callGroq } from "./groq-client";

export type QuizQuestion = {
  question: string;
  options: string[];
  correctIndex: number;
  explanation: string;
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
  executiveSummary: z.string(),
  keyInsights: z.array(z.object({ title: z.string(), body: z.string() })),
  chapters: z.array(z.object({ title: z.string(), summary: z.string() })),
});

const outputSchema = z.object({
  questions: z
    .array(
      z.object({
        question: z.string(),
        options: z.array(z.string()).length(4),
        correctIndex: z.number().int().min(0).max(3),
        explanation: z.string().default(""),
      }),
    )
    .min(1),
});

export const generateQuiz = createServerFn({ method: "POST" })
  .inputValidator((i: unknown) =>
    z.object({ context: contextSchema }).parse(i),
  )
  .handler(async ({ data }): Promise<{ questions: QuizQuestion[] }> => {
    const ctx = `Video: ${data.context.title}
Summary:
${data.context.executiveSummary}
Insights:
${data.context.keyInsights.map((k) => `- ${k.title}: ${k.body}`).join("\n")}
Chapters:
${data.context.chapters.map((c) => `- ${c.title}: ${c.summary}`).join("\n")}`;

    const content = await callGroq({
      models: ["openai/gpt-oss-20b", "openai/gpt-oss-120b"],
      keyOffset: 0,
      messages: [
        {
          role: "system",
          content: `Generate a 5-question multiple-choice quiz that tests real comprehension of the video below. Rules: each question must have exactly 4 plausible options and exactly one correct answer. Vary correctIndex (0-3) across questions. Include a short explanation. Return JSON only in this shape: {"questions":[{"question":string,"options":[string,string,string,string],"correctIndex":number,"explanation":string}]}`,
        },
        { role: "user", content: ctx },
      ],
      maxTokens: 1200,
      temperature: 0.5,
    });

    let parsed: unknown;
    try {
      parsed = JSON.parse(extractJson(content));
    } catch {
      throw new Error("AI returned malformed quiz output.");
    }
    return outputSchema.parse(parsed);
  });
