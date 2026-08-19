import { createServerFn } from "@tanstack/react-start";
import { z } from "zod";
import type { LearningReport, SkipSegmentKind } from "./report-data";
import { callGroq } from "./groq-client";

function extractVideoId(url: string): string | null {
  try {
    const u = new URL(url.startsWith("http") ? url : `https://${url}`);
    const host = u.hostname.replace(/^www\./, "");
    if (host === "youtu.be") return u.pathname.slice(1).split("/")[0] || null;
    if (host.endsWith("youtube.com")) {
      if (u.pathname === "/watch") return u.searchParams.get("v");
      const parts = u.pathname.split("/").filter(Boolean);
      if (["shorts", "embed", "v"].includes(parts[0])) return parts[1] ?? null;
    }
    return null;
  } catch {
    return null;
  }
}

function extractJson(raw: string): string {
  let s = raw.trim();
  s = s.replace(/^```(?:json)?\s*/i, "").replace(/```\s*$/i, "").trim();
  const first = s.search(/[{[]/);
  if (first === -1) throw new Error("no json");
  const openCh = s[first];
  const closeCh = openCh === "{" ? "}" : "]";
  const last = s.lastIndexOf(closeCh);
  if (last <= first) throw new Error("no json");
  return s.slice(first, last + 1);
}

interface TranscriptSegment {
  start: number;
  dur: number;
  text: string;
}

async function fetchTranscript(videoId: string): Promise<TranscriptSegment[]> {
  const apiKey = process.env.TRANSCRIPT_API_KEY;
  if (!apiKey) {
    throw new Error(
      "Transcript service key missing. Please add TRANSCRIPT_API_KEY to enable analysis.",
    );
  }

  const res = await fetch("https://www.youtube-transcript.io/api/transcripts", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Basic ${apiKey}`,
    },
    body: JSON.stringify({ ids: [videoId] }),
  });

  if (!res.ok) {
    if (res.status === 429) throw new Error("Transcript service is busy right now. Please try again in a moment.");
    throw new Error("Could not load this video. Please try another URL.");
  }

  const json = (await res.json()) as Array<{
    id?: string;
    tracks?: Array<{
      language?: string;
      transcript?: Array<{ text?: string; start?: string | number; dur?: string | number }>;
    }>;
  }>;

  const entry = json.find((e) => e.id === videoId) ?? json[0];
  const track =
    entry?.tracks?.find((t) => t.language?.toLowerCase().startsWith("en")) ??
    entry?.tracks?.[0];

  if (!track?.transcript?.length) {
    throw new Error("This video is private, unavailable, or has no transcript.");
  }

  const segs = track.transcript
    .map((t) => ({
      start: Number(t.start ?? 0),
      dur: Number(t.dur ?? 0),
      text: (t.text ?? "").replace(/\s+/g, " ").trim(),
    }))
    .filter((s) => s.text);

  if (!segs.length) throw new Error("Transcript unavailable for this video.");
  return segs;
}

const reportSchema = z.object({
  title: z.string(),
  channel: z.string().default("YouTube"),
  category: z.string().default("Education"),
  language: z.string().default("English"),
  worthWatching: z.enum(["Yes", "Skim", "No"]),
  overallScore: z.number().min(0).max(10),
  scoreExplanation: z.string(),
  scoreBreakdown: z
    .array(z.object({ label: z.string(), score: z.number().min(0).max(10) }))
    .length(7),
  executiveSummary: z.string(),
  keyInsights: z.array(z.object({ title: z.string(), body: z.string().default("") })).min(5).max(7),
  chapters: z
    .array(z.object({ title: z.string(), start: z.number().min(0), summary: z.string().default("") }))
    .min(1),
  skipMap: z
    .array(
      z.object({
        kind: z.enum(["watch", "optional", "skip"]),
        label: z.string(),
        start: z.number().min(0),
        end: z.number().min(0),
        reason: z.string().default(""),
        isBestMoment: z.boolean().default(false),
      }),
    )
    .min(1),
  videoDna: z
    .array(z.object({ label: z.string(), percentage: z.number().min(0).max(100) }))
    .min(4)
    .max(6),
});

function sanitizeTimestamps(
  text: string,
  validSeconds: number[],
  durationSec: number,
): string {
  if (!validSeconds.length) return text;
  return text.replace(/\b(\d{1,3}):([0-5]\d)\b/g, (match, mStr, sStr) => {
    const m = parseInt(mStr, 10);
    const s = parseInt(sStr, 10);
    const totalSec = Math.min(m * 60 + s, durationSec);
    let nearest = validSeconds[0];
    let bestDiff = Math.abs(totalSec - nearest);
    for (const v of validSeconds) {
      const diff = Math.abs(totalSec - v);
      if (diff < bestDiff) {
        bestDiff = diff;
        nearest = v;
      }
    }
    const nm = Math.floor(nearest / 60);
    const ns = nearest % 60;
    return `${nm}:${String(ns).padStart(2, "0")}`;
  });
}

// A full analysis request (system prompt + sampled transcript + reserved
// completion tokens) runs ~6,500 tokens on llama-3.3-70b-versatile, which
// has a 12,000 TPM / 100,000 TPD budget on the free tier — the only model
// with enough headroom for this call size (llama-3.1-8b-instant's 6,000 TPM
// budget is too small for it). See groq-client.ts for key rotation, which
// is the real lever for scaling total daily capacity beyond one key's cap.
const MAX_COMPLETION_TOKENS = 3000;
const MAX_TRANSCRIPT_CHARS = 9000;

// Instead of slicing only the START of the transcript (which meant a 3hr
// video was only ever analyzed on its first ~20 minutes), sample evenly
// spaced windows across the FULL duration so every video — regardless of
// length — gets proportional coverage of its beginning, middle, and end.
function sampleTranscript(full: string, maxChars: number): { text: string; sampled: boolean } {
  if (full.length <= maxChars) return { text: full, sampled: false };

  const lines = full.split("\n");
  const totalChars = full.length;
  const windowCount = 8;
  const charsPerWindow = Math.floor(maxChars / windowCount);

  const picked: string[] = [];
  for (let w = 0; w < windowCount; w++) {
    const targetCharStart = Math.floor((totalChars / windowCount) * w);
    let acc = 0;
    let startLine = 0;
    for (let i = 0; i < lines.length; i++) {
      if (acc >= targetCharStart) {
        startLine = i;
        break;
      }
      acc += lines[i].length + 1;
    }
    let windowText = "";
    let i = startLine;
    while (i < lines.length && windowText.length < charsPerWindow) {
      windowText += lines[i] + "\n";
      i++;
    }
    if (windowText) picked.push(windowText.trim());
  }

  return { text: picked.join("\n...\n"), sampled: true };
}

// AI-estimated percentages rarely sum to exactly 100 due to independent
// per-category estimation. This guarantees an exact 100% total by scaling
// proportionally, then distributing any rounding remainder to the
// categories with the largest fractional part — never left to chance.
function normalizeVideoDna(
  items: { label: string; percentage: number }[],
): { label: string; percentage: number }[] {
  const total = items.reduce((sum, i) => sum + i.percentage, 0);
  if (total <= 0) return items.map((i) => ({ ...i, percentage: 0 }));

  const scaled = items.map((i) => {
    const raw = (i.percentage / total) * 100;
    return { label: i.label, percentage: Math.floor(raw), remainder: raw - Math.floor(raw) };
  });

  const currentSum = scaled.reduce((sum, i) => sum + i.percentage, 0);
  let remaining = 100 - currentSum;

  const byRemainder = [...scaled].sort((a, b) => b.remainder - a.remainder);
  for (let i = 0; i < remaining; i++) {
    byRemainder[i % byRemainder.length].percentage += 1;
  }

  return scaled.map((s) => ({ label: s.label, percentage: s.percentage }));
}

async function generateReport(args: {
  url: string;
  videoId: string;
  transcript: string;
  durationSec: number;
}): Promise<LearningReport> {
  const system = `You are WisTube AI, an expert learning analyst. Analyze the transcript of a YouTube video and produce a rigorous Learning Report as JSON. The transcript may be in ANY language — always understand it correctly regardless of language, but ALWAYS write the entire report (title, executiveSummary, scoreExplanation, keyInsights, chapters, skipMap reasons — everything) in English, regardless of what language the video itself is in. Set the "language" field to the video's actual detected language (e.g. "Hindi", "Spanish"), even though your written report stays in English. Be honest — if the video is thin or filler-heavy, say so. This video is exactly ${args.durationSec} seconds long (${Math.floor(args.durationSec / 60)}:${String(args.durationSec % 60).padStart(2, "0")}). All numeric "start"/"end" fields (in chapters and skipMap) are in SECONDS and MUST be between 0 and ${args.durationSec} — never exceed this. CRITICAL: whenever you refer to a time in PROSE TEXT (executiveSummary, scoreExplanation, keyInsights, reason fields), you MUST write it as mm:ss (e.g. "5:37"), and that time MUST be less than or equal to ${Math.floor(args.durationSec / 60)}:${String(args.durationSec % 60).padStart(2, "0")} (the video's actual length). NEVER invent or estimate a timestamp — only reference times that correspond to an actual moment in the transcript provided below. Do not write a prose timestamp higher than the video's total duration under any circumstance. Chapters must be in chronological order. Skip Map segments must cover the whole video contiguously (start=0, last end=${args.durationSec}, each segment.start = previous.end). Return JSON only, matching this shape exactly:
{
  "title": string,               // best guess of the video's title/topic
  "channel": string,             // best guess of channel/creator, or "Unknown"
  "category": string,            // e.g. "Education", "Productivity"
  "language": string,
  "worthWatching": "Yes"|"Skim"|"No",
  "overallScore": number,        // 0-10 with one decimal (e.g. 8.9)
  "scoreExplanation": string,    // 2-3 sentences
  "scoreBreakdown": [ {"label": string, "score": number} ], // EXACTLY 7 items with these exact labels, each scored 0-10: "Clarity", "Depth", "Structure", "Practical Value", "Examples", "Evidence & Accuracy", "Learning Efficiency"
  "executiveSummary": string,    // 3-5 sentences, mention strongest section with timestamps
  "keyInsights": [ {"title": string, "body": string} ], // 5-7 items. "title" MUST be a complete, self-contained one-line takeaway a reader can scan in 5 seconds (e.g. "Success comes from perseverance, not luck") — not a short label. "body" can stay brief, used for context elsewhere but not the main display.
  "chapters": [ {"title": string, "start": number, "summary": string} ], // 4-7 items
  "skipMap": [ {"kind": "watch"|"optional"|"skip", "label": string, "start": number, "end": number, "reason": string, "isBestMoment": boolean} ], // 4-6 items covering the video. EXACTLY ONE segment across the whole array must have isBestMoment=true. CRITICAL: evaluate each segment's actual content independently — do NOT default to picking the 3rd segment, the middle segment, or any fixed position out of habit. The best moment could be the 1st, 2nd, last, or any segment — base the choice purely on which segment has the single highest concentration of valuable insight/information density in THIS SPECIFIC video''s transcript. All other segments must have isBestMoment=false.
  "videoDna": [ {"label": string, "percentage": number} ] // Break the ENTIRE video's runtime into these exact 6 categories, percentages must sum to 100: "Core Concepts", "Examples", "Stories", "Repetition", "Sponsor/Promotion", "Filler". Estimate honestly based on the transcript's actual content mix — do not default to even splits.
}`;

  const { text: transcriptForPrompt, sampled } = sampleTranscript(
    args.transcript,
    MAX_TRANSCRIPT_CHARS,
  );

  const user = `Video URL: ${args.url}
Video duration: ${args.durationSec} seconds.

Transcript (each line prefixed with [start_seconds]${sampled ? ", sampled evenly across the full video — sections are separated by '...' markers; infer the connecting content between samples proportionally" : ""}):
${transcriptForPrompt}

Return the JSON report now.`;

  const content = await callGroq({
       models: ["openai/gpt-oss-120b"],
    messages: [
      { role: "system", content: system },
      { role: "user", content: user },
    ],
    maxTokens: MAX_COMPLETION_TOKENS,
    temperature: 0.2,
  });

  let parsed: unknown;
  try {
    parsed = JSON.parse(extractJson(content));
  } catch {
    throw new Error("AI returned malformed output.");
  }
  if (Array.isArray(parsed)) {
    parsed = parsed[0];
  }
  const r = reportSchema.parse(parsed);

  const timeSavedSec = r.skipMap.reduce((acc, s) => {
    const len = Math.max(0, s.end - s.start);
    if (s.kind === "skip") return acc + len;
    if (s.kind === "optional") return acc + len * 0.5;
    return acc;
  }, 0);

  const clampedChapters = r.chapters
    .map((c, i) => ({
      id: `c${i}`,
      title: c.title,
      start: Math.min(Math.max(0, Math.floor(c.start)), args.durationSec),
      summary: c.summary,
    }))
    .sort((a, b) => a.start - b.start);

  const clampedSkip = r.skipMap.map((s, i) => ({
    id: `s${i}`,
    kind: s.kind as SkipSegmentKind,
    label: s.label,
    start: Math.min(Math.max(0, Math.floor(s.start)), args.durationSec),
    end: Math.min(Math.max(0, Math.floor(s.end)), args.durationSec),
    reason: s.reason,
    isBestMoment: s.isBestMoment,
  }));

  const validSeconds = Array.from(
    new Set([
      0,
      args.durationSec,
      ...clampedChapters.map((c) => c.start),
      ...clampedSkip.flatMap((s) => [s.start, s.end]),
    ]),
  ).sort((a, b) => a - b);

  const sanitize = (t: string) => sanitizeTimestamps(t, validSeconds, args.durationSec);

  const sanitizedChapters = clampedChapters.map((c) => ({
    ...c,
    summary: sanitize(c.summary),
  }));
  const sanitizedSkip = clampedSkip.map((s) => ({
    ...s,
    reason: sanitize(s.reason),
  }));

  return {
    videoId: args.videoId,
    url: args.url,
    title: r.title,
    channel: r.channel,
    category: r.category,
    language: r.language,
    durationSec: args.durationSec,
    timeSavedSec: Math.floor(timeSavedSec),
    worthWatching: r.worthWatching,
    overallScore: Math.round(r.overallScore * 10) / 10,
    scoreExplanation: sanitize(r.scoreExplanation),
    scoreBreakdown: r.scoreBreakdown.map((b) => ({
      label: b.label,
      score: Math.round(b.score * 10) / 10,
    })),
    executiveSummary: sanitize(r.executiveSummary),
    keyInsights: r.keyInsights.map((k) => ({
      title: k.title,
      body: sanitize(k.body),
    })),
    chapters: sanitizedChapters,
    skipMap: sanitizedSkip,
    videoDna: normalizeVideoDna(r.videoDna),
  };
}

export const analyzeVideo = createServerFn({ method: "POST" })
  .inputValidator((input: unknown) => z.object({ url: z.string().min(1) }).parse(input))
  .handler(async ({ data }): Promise<LearningReport> => {
    const videoId = extractVideoId(data.url);
    if (!videoId) throw new Error("That doesn't look like a valid YouTube URL.");
    const transcript = await fetchTranscript(videoId);
    const last = transcript[transcript.length - 1];
    const durationSec = Math.max(60, Math.ceil(last.start + (last.dur || 0)));
    const transcriptText = transcript
      .map((t) => `[${Math.floor(t.start)}] ${t.text}`)
      .join("\n");
    return generateReport({
      url: data.url,
      videoId,
      transcript: transcriptText,
      durationSec,
    });
  });
