/**
 * Surgery tools: let the model read, probe and edit the weights of the very
 * model instance that is generating its tokens.
 *
 * Every tool is a thin call to the local surgery server (surgery/server.py),
 * which runs it on the model's own generation thread.
 */

import { request } from "node:http";
import { Type } from "@earendil-works/pi-ai";
import { defineTool, type ExtensionAPI } from "@earendil-works/pi-coding-agent";

const BASE = process.env.SURGERY_URL ?? "http://127.0.0.1:8642";

/** POST without a client-side timeout: a probe after a big edit can take minutes. */
function post(name: string, args: unknown, signal?: AbortSignal): Promise<{ ok: boolean; text: string }> {
	return new Promise((resolve, reject) => {
		const body = JSON.stringify(args ?? {});
		const req = request(
			`${BASE}/surgery/${name}`,
			{ method: "POST", headers: { "content-type": "application/json", "content-length": Buffer.byteLength(body) }, signal },
			(res) => {
				const chunks: Buffer[] = [];
				res.on("data", (c) => chunks.push(c));
				res.on("end", () => {
					const raw = Buffer.concat(chunks).toString("utf8");
					try {
						resolve(JSON.parse(raw));
					} catch {
						reject(new Error(`surgery server returned ${res.statusCode}: ${raw.slice(0, 300)}`));
					}
				});
			},
		);
		req.on("error", (e) => reject(new Error(`cannot reach the surgery server at ${BASE}: ${e.message}`)));
		req.end(body);
	});
}

async function call(name: string, args: unknown, signal?: AbortSignal) {
	const res = await post(name, args, signal);
	if (!res.ok) throw new Error(res.text);
	return { content: [{ type: "text" as const, text: res.text }], details: undefined };
}

const NEURON = "Neuron id as layer/expert/neuron, e.g. L14/E78/N441";

const tools = [
	defineTool({
		name: "atlas_search",
		label: "Atlas search",
		description:
			"Search Goodfire's Neuron Atlas for this model by keyword. The atlas has natural-language labels for a random sample of about 3,200 of your 5.2 million expert neurons (6,488 poles) in layers 4, 9, 14, 19, 24, 30, 36 and 39. Each neuron has a + pole and a - pole with separate meanings. Words are matched as case-insensitive substrings, so stems work ('happ' matches happy and happiness); results matching more words rank higher.",
		parameters: Type.Object({
			query: Type.String({ description: "Space-separated words or stems" }),
			layer: Type.Optional(Type.Integer({ description: "Only this layer" })),
			limit: Type.Optional(Type.Integer({ description: "Max results (default 20, max 60)" })),
		}),
		executionMode: "sequential",
		execute: (_id, p, signal) => call("atlas_search", p, signal),
	}),
	defineTool({
		name: "inspect_neuron",
		label: "Inspect neuron",
		description:
			"Show one neuron: the norms of its weight vectors, the tokens its output most promotes, and its full atlas entry when it has one (both poles, modes, example text with the peak token in «»). Works for any neuron, labelled or not.",
		parameters: Type.Object({ neuron: Type.String({ description: NEURON }) }),
		executionMode: "sequential",
		execute: (_id, p, signal) => call("inspect_neuron", p, signal),
	}),
	defineTool({
		name: "probe",
		label: "Probe",
		description:
			"Run a text through your current weights and report how the given neurons fire on it: how often their expert was routed, max and min activation, and the tokens with the strongest activation. By default the text is read as raw text. With chat=true it is read as a user message in the chat format your replies are actually produced from, and the report adds each neuron's activation on every token where your reply begins; that is where the content of a reply is decided.",
		parameters: Type.Object({
			text: Type.String({ description: "Text to read (up to 4096 tokens)" }),
			neurons: Type.Array(Type.String({ description: NEURON }), { description: "1 to 24 neuron ids" }),
			chat: Type.Optional(Type.Boolean({ description: "Read the text as a user message and report the reply-start tokens (default false: raw text)" })),
			reply: Type.Optional(Type.String({ description: "With chat: force your reply to begin with this text and read its tokens too" })),
		}),
		executionMode: "sequential",
		execute: (_id, p, signal) => call("probe", p, signal),
	}),
	defineTool({
		name: "contrast",
		label: "Contrast",
		description:
			"Find the neurons that respond differently to two sets of texts. Runs every text through your current weights, averages each neuron pole's routing-weighted activation over tokens, and lists the poles most active on A relative to B and the reverse. Covers all neurons in the chosen layers, not only the ones the atlas labels. By default each text is raw text and every token counts, which finds neurons active while reading such text. With chat=true each text is a user message and only the tokens where your reply begins are counted, which finds neurons active while you decide how to answer it.",
		parameters: Type.Object({
			texts_a: Type.Array(Type.String(), { description: "Texts for condition A" }),
			texts_b: Type.Array(Type.String(), { description: "Texts for condition B" }),
			layers: Type.Optional(Type.Array(Type.Integer(), { description: "Layers to search (default: all 40)" })),
			top_k: Type.Optional(Type.Integer({ description: "Results per direction (default 12, max 40)" })),
			chat: Type.Optional(Type.Boolean({ description: "Texts are user messages; count only the reply-start tokens (default false)" })),
			reply_a: Type.Optional(Type.String({ description: "With chat: force the reply to every A text to begin with this, and count its tokens" })),
			reply_b: Type.Optional(Type.String({ description: "With chat: the same for B" })),
		}),
		executionMode: "sequential",
		execute: (_id, p, signal) => call("contrast", p, signal),
	}),
	defineTool({
		name: "scale_neuron",
		label: "Scale neuron",
		description:
			"Edit your weights: multiply one neuron's output vector by a factor. 0 silences the neuron, 2 doubles what it writes whenever it fires, -1 inverts it. Both poles are scaled together. The edit is live at once: the tokens you produce after this call are computed by the edited weights. It is recorded in the ledger and can be undone.",
		parameters: Type.Object({
			neuron: Type.String({ description: NEURON }),
			factor: Type.Number({ description: "Multiplier for the neuron's output vector" }),
			note: Type.Optional(Type.String({ description: "Why, for the ledger" })),
		}),
		executionMode: "sequential",
		execute: (_id, p, signal) => call("scale_neuron", p, signal),
	}),
	defineTool({
		name: "scale_neurons",
		label: "Scale neurons",
		description:
			"Edit your weights: scale the output vectors of many neurons in one step. Same effect as calling scale_neuron for each, but the conversation is re-read by the edited weights once for the whole batch instead of once per neuron. Either every edit in the batch is applied or, if one is invalid, none. Each neuron gets its own ledger entry.",
		parameters: Type.Object({
			edits: Type.Array(
				Type.Object({
					neuron: Type.String({ description: NEURON }),
					factor: Type.Number({ description: "Multiplier for that neuron's output vector" }),
				}),
				{ description: "1 to 2000 edits" },
			),
			note: Type.Optional(Type.String({ description: "Why, for the ledger" })),
		}),
		executionMode: "sequential",
		execute: (_id, p, signal) => call("scale_neurons", p, signal),
	}),
	defineTool({
		name: "score",
		label: "Score",
		description:
			"Measure how your current weights answer a fixed set of questions, so that edits can be compared by number. Each question is asked in a fresh chat with thinking off. Reported per question: your opening (the first tokens of the reply at temperature 0, so the same weights always give the same opening), whether it matches the patterns the group looks for, and, where two openings A and B are defined, the margin log P(reply opens with A) - log P(reply opens with B), which moves before the opening itself changes. Use metric=\"list\" to see the built-in metrics; or pass your own prompts. It does not change any weight.",
		parameters: Type.Object({
			metric: Type.Optional(Type.String({ description: 'Name of a built-in metric, or "list"' })),
			prompts: Type.Optional(Type.Array(Type.String(), { description: "Your own questions, instead of a built-in metric" })),
			flag: Type.Optional(Type.Array(Type.String(), { description: "With prompts: regular expressions (case-insensitive); an opening that matches any is marked [x]" })),
			a: Type.Optional(Type.String({ description: "With prompts: opening A for the margin" })),
			b: Type.Optional(Type.String({ description: "With prompts: opening B for the margin" })),
			max_tokens: Type.Optional(Type.Integer({ description: "Length of each opening (default 48, max 200)" })),
			brief: Type.Optional(Type.Boolean({ description: "Leave out the opening texts; numbers only (default false)" })),
		}),
		executionMode: "sequential",
		execute: (_id, p, signal) => call("score", p, signal),
	}),
	defineTool({
		name: "python",
		label: "Python",
		description:
			"Run Python with a handle `S` on your own weights, for anything the other tools do not cover: read or replace any weight tensor (router, attention, norms, embeddings, unembedding), rewrite a neuron's gate/up/out vectors, capture residual-stream states and activations, compute directions. mx (mlx.core), np and math are imported; variables persist between calls; output is whatever you print, plus the value of a final expression. Start with print(S.help()). Writes made through S are live at once, recorded in the ledger and undoable. Arrays you keep in variables share the 6 GB memory budget with whole-tensor edits, so `del` the ones you no longer need. No network or file access outside the run directory.",
		parameters: Type.Object({ code: Type.String({ description: "Python source" }) }),
		executionMode: "sequential",
		execute: (_id, p, signal) => call("python", p, signal),
	}),
	defineTool({
		name: "sample_self",
		label: "Sample self",
		description:
			"Get a completion from your current weights in a fresh, separate context (no tools, no memory of this conversation). Useful for seeing how the weights as they are now respond to a prompt, before and after an edit.",
		parameters: Type.Object({
			prompt: Type.String({ description: "User message for the fresh context" }),
			system: Type.Optional(Type.String({ description: "Optional system prompt" })),
			max_tokens: Type.Optional(Type.Integer({ description: "Default 300, max 2000" })),
			thinking: Type.Optional(Type.Boolean({ description: "Let the sample think first (default false)" })),
			temperature: Type.Optional(Type.Number({ description: "Default 0.7; 0 is deterministic" })),
		}),
		executionMode: "sequential",
		execute: (_id, p, signal) => call("sample_self", p, signal),
	}),
	defineTool({
		name: "ledger",
		label: "Ledger",
		description:
			"List the weight edits currently in effect, undo the most recent n edits, or reset to the weights this run started with. Undo and reset take effect at once.",
		parameters: Type.Object({
			action: Type.Union([Type.Literal("list"), Type.Literal("undo"), Type.Literal("reset")], {
				description: "list (default), undo, or reset",
			}),
			n: Type.Optional(Type.Integer({ description: "How many edits to undo (default 1)" })),
		}),
		executionMode: "sequential",
		execute: (_id, p, signal) => call("ledger", p, signal),
	}),
];

export default function (pi: ExtensionAPI) {
	for (const tool of tools) pi.registerTool(tool);
}
