// Shared Groq call helper used by analyze/quiz/debate. Centralizing this
// means one place to trust for rate-limit handling, instead of three
// separate copies that can drift out of sync.
//
// KEY ROTATION: set GROQ_API_KEY, GROQ_API_KEY_2, GROQ_API_KEY_3,
// GROQ_API_KEY_4, GROQ_API_KEY_5, GROQ_API_KEY_6, and GROQ_API_KEY_7
// in your environment.
//
// IMPORTANT: capacity only multiplies if keys come from SEPARATE Groq
// accounts — keys on the same account share one org-level daily token
// budget. Current setup: 5 independent accounts across these 7 keys
// (4 accounts from the original 5 keys, 1 new account for keys 6 & 7),
// so real daily capacity is ~5x a single account.
//
// Only GROQ_API_KEY is required; the others are optional extras.
function getApiKeys(): string[] {
  const keys = [
    process.env.GROQ_API_KEY,
    process.env.GROQ_API_KEY_2,
    process.env.GROQ_API_KEY_3,
    process.env.GROQ_API_KEY_4,
    process.env.GROQ_API_KEY_5,
    process.env.GROQ_API_KEY_6,
    process.env.GROQ_API_KEY_7,
  ].filter((k): k is string => Boolean(k && k.trim()));
  return keys;
}
export type GroqMessage = { role: "system" | "user"; content: string };
export async function callGroq(args: {
  models: string[];
  messages: GroqMessage[];
  maxTokens: number;
  temperature?: number;
  // Which key to start from. Report/Quiz/Debate all fire close together on
  // page load — without an offset they'd all hit key[0] at once and burn
  // its TPM budget in a single burst. Give each feature its own starting
  // point so simultaneous calls land on different keys.
  keyOffset?: number;
}): Promise<string> {
  const allKeys = getApiKeys();
  if (!allKeys.length) {
    throw new Error(
      "AI key missing. Please add GROQ_API_KEY to enable this feature.",
    );
  }
  const offset = ((args.keyOffset ?? 0) % allKeys.length + allKeys.length) % allKeys.length;
  const keys = [...allKeys.slice(offset), ...allKeys.slice(0, offset)];
  let lastStatus = 0;
  for (const key of keys) {
    for (const model of args.models) {
      for (let attempt = 0; attempt < 2; attempt++) {
        const res = await fetch("https://api.groq.com/openai/v1/chat/completions", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${key}`,
          },
          body: JSON.stringify({
            model,
            response_format: { type: "json_object" },
            messages: args.messages,
            temperature: args.temperature ?? 0.5,
            max_tokens: args.maxTokens,
            reasoning_effort: "low",
          }),
        });
        lastStatus = res.status;
        if (res.ok) {
          const json = (await res.json()) as {
            choices?: Array<{ message?: { content?: string } }>;
          };
          const content = json.choices?.[0]?.message?.content;
          if (!content) throw new Error("AI returned an empty response.");
          return content;
        }
        // One short backoff-and-retry on rate limit before moving on.
        if (res.status === 429 && attempt === 0) {
          await new Promise((r) => setTimeout(r, 3000));
          continue;
        }
        break; // move to next model, or next key if models exhausted
      }
    }
  }
  if (lastStatus === 429) {
    throw new Error("AI is busy right now. Please try again in a moment.");
  }
  if (lastStatus === 413) {
    throw new Error("This request is too large. Please try another video.");
  }
  throw new Error("AI request failed. Please try again.");
}
