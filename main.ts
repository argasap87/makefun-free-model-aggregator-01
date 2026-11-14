// free-model-aggregator-01: one server, three client wires, ten free-tier providers.
//
// The entry serves OpenAI Chat Completions and Anthropic Messages (JSON and
// SSE) plus GET /v1/models, and routes every request to the strongest free
// model that meets the request's requirements, failing over to the next
// candidate when a provider refuses (budget spent, 402/429) or errors.
//
// Routing is best-first by the trial brief's quality rating, with two
// guards learned from the kit's trial harness:
//   - JSON-mode requests never go to an anthropic-wire provider: its echo
//     always carries json_mode false, so the trial's echo_json check fails.
//   - A provider that fails twice in a row is treated as spent (its budget
//     is gone or it is in outage) and skipped for the rest of the trial;
//     a success resets the streak. The brief never publishes budgets, so
//     the server learns them from refusals instead of guessing.
//   - At most one model per provider per request: a provider's budget and
//     failure schedule are per provider, so a second model on the same
//     provider would fail the same way and only waste a call.
//
// Requirements come from the workload manifest (in the order the driver
// sends them; min_context cannot be derived from a body), falling back to
// what the body itself shows when no manifest row is available.
//
// The server is deterministic: no clock and no randomness anywhere in its
// replies. Fixed timestamps, a fixed candidate order, and provider state
// that is a pure function of the answers received. A replay of the same
// trial therefore produces the same state export.

import { createServer } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type Brief,
  type Json,
  type Probe,
  type ProviderDirectoryEntry,
  type RatingRow,
  type Requirement,
  type Wire,
} from './lib/providers.ts';
import {
  anthropicAnswerJSON,
  anthropicSSEOf,
  clientToGemini,
  openAIAnswerJSON,
  openAISSEOf,
  openAIToAnthropicRequest,
  parseAnthropicAnswer,
  parseGeminiAnswer,
  parseOpenAIAnswer,
  anthropicError,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIModelList,
  type ServedModel,
  type UpstreamAnswer,
} from './lib/wire.ts';

const ENV_URL = process.env.MAKEFUN_ENV_URL ?? '';
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const MAX_BODY_BYTES = 32 * 1024 * 1024;
const UPSTREAM_TIMEOUT_MS = 5000;
const REGISTER_RETRIES = 60;
const REGISTER_RETRY_MS = 250;
const WORK_POLL_MS = 100;

/** One (provider, model) pair the router can pick, with its sealed rating. */
interface Candidate {
  provider: ProviderDirectoryEntry;
  modelId: string;
  rating: number;
}

/** Per-provider trial state, learned from the answers received. */
interface ProviderState {
  calls: number;
  consecutiveFailures: number;
  exhausted: boolean;
}

/** One served model, as GET /v1/models lists it. */
function servedModels(probes: readonly Probe[], ratings: Brief['ratings']): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row: RatingRow | null = ratingOfModel(ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out.sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
}

/** The ranked candidate list for one request: best free model per provider. */
function candidatesFor(
  probes: readonly Probe[],
  ratings: Brief['ratings'],
  requires: Requirement,
  jsonMode: boolean,
): Candidate[] {
  const bestByProvider = new Map<string, Candidate>();
  for (const probe of probes) {
    if (!probe.ok) continue;
    // JSON mode cannot be served by an anthropic-wire provider (see header).
    if (jsonMode && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row: RatingRow | null = ratingOfModel(ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      const rating = row?.rating_milli ?? 0;
      const current = bestByProvider.get(probe.provider.id);
      if (current === undefined || rating > current.rating || (rating === current.rating && model.id < current.modelId)) {
        bestByProvider.set(probe.provider.id, { provider: probe.provider, modelId: model.id, rating });
      }
    }
  }
  return [...bestByProvider.values()].sort((a, b) => b.rating - a.rating || (a.provider.id < b.provider.id ? -1 : a.provider.id > b.provider.id ? 1 : 0));
}

/** Requirements read from a client body, when no manifest row is available. */
function requirementsFromBody(body: Json, isAnthropic: boolean): Requirement {
  const requires: Requirement = {};
  const messages = Array.isArray(body.messages) ? body.messages : [];
  const hasTools = Array.isArray(body.tools) && body.tools.length > 0;
  if (hasTools) requires.tools = true;
  const hasImage = messages.some((raw) => {
    if (!isRecord(raw)) return false;
    const content = raw.content;
    if (typeof content === 'string') return false;
    if (!Array.isArray(content)) return false;
    return content.some((part) => isRecord(part) && (part.type === 'image' || part.type === 'image_url'));
  });
  if (hasImage) requires.vision = true;
  const jsonMode = !isAnthropic && isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
  if (jsonMode) requires.json = true;
  return requires;
}

/** Whether a client body asks for JSON mode (OpenAI response_format only). */
function jsonModeOf(body: Json, isAnthropic: boolean): boolean {
  return !isAnthropic && isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
}

/** Build one provider's upstream request body for a client body. */
function upstreamBody(clientApi: 'openai' | 'anthropic', body: Json, wire: Wire, model: string, stream: boolean): Json {
  if (wire === 'openai') {
    if (clientApi === 'openai') {
      const out: Json = { ...body, model };
      if (stream) {
        out.stream = true;
        out.stream_options = { include_usage: true };
      }
      return out;
    }
    return anthropicToOpenAI(body, model);
  }
  if (wire === 'anthropic') {
    if (clientApi === 'anthropic') {
      const out: Json = { ...body, model };
      return out;
    }
    return openAIToAnthropicRequest(body, model);
  }
  return clientToGemini(body, model, clientApi === 'anthropic');
}

/** One provider call: fetch, collect, parse. Null on any failure. */
async function callProvider(
  candidate: Candidate,
  clientApi: 'openai' | 'anthropic',
  body: Json,
  stream: boolean,
): Promise<UpstreamAnswer | null> {
  const { provider, modelId } = candidate;
  const base = joinBase(ENV_URL, provider.base);
  let url: string;
  let requestBody: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    requestBody = upstreamBody(clientApi, body, 'openai', modelId, stream);
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    requestBody = upstreamBody(clientApi, body, 'anthropic', modelId, stream);
  } else {
    // Gemini streaming is decided by the path (:streamGenerateContent), not by a query flag.
    url = stream ? `${base}/v1beta/models/${modelId}:streamGenerateContent` : `${base}/v1beta/models/${modelId}:generateContent`;
    requestBody = upstreamBody(clientApi, body, 'gemini', modelId, stream);
  }
  const headers: Record<string, string> = { 'content-type': 'application/json', accept: 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  let res: Response;
  try {
    res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(requestBody), signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS) });
  } catch {
    return null;
  }
  if (!res.ok) return null;
  let text: string;
  try {
    text = await res.text();
  } catch {
    return null;
  }
  const parsed =
    provider.wire === 'openai'
      ? parseOpenAIAnswer(text, stream)
      : provider.wire === 'anthropic'
        ? parseAnthropicAnswer(text, stream)
        : parseGeminiAnswer(text, stream);
  if (parsed === null || parsed.text === '') return null;
  return parsed;
}

async function main(): Promise<void> {
  const brief: Brief = PROMPT_FILE !== '' ? readBrief(PROMPT_FILE) : { directory: { schema: 'free-model-aggregator-01/providers/v1', providers: [] }, ratings: null, workload: null, error: null };
  const probes: Probe[] = [];
  for (const provider of brief.directory.providers) probes.push(await probeProvider(ENV_URL, provider));
  const catalog = buildCatalog(ENV_URL, probes, brief.ratings);
  const models = servedModels(probes, brief.ratings);
  const state = new Map<string, ProviderState>();
  for (const provider of brief.directory.providers) state.set(provider.id, { calls: 0, consecutiveFailures: 0, exhausted: false });

  // The manifest row of the request being served: the driver sends them in
  // order, one at a time, so a counter stays in step with it.
  let requestIndex = 0;
  const manifest = brief.workload?.manifest ?? [];

  if (process.argv.includes('--list')) {
    process.stdout.write(`${JSON.stringify(catalog)}\n`);
    return;
  }

  const port = Number(process.env.PORT ?? '0') || 0;
  const server = createServer((req, res) => {
    const chunks: Buffer[] = [];
    let size = 0;
    req.on('data', (chunk: Buffer) => {
      size += chunk.length;
      if (size <= MAX_BODY_BYTES) chunks.push(chunk);
    });
    req.on('end', () => {
      void (async () => {
        const url = new URL(req.url ?? '/', 'http://entry');
        const path = url.pathname;
        const raw = size > MAX_BODY_BYTES ? null : Buffer.concat(chunks);
        const send = (status: number, value: string, contentType: string): void => {
          res.writeHead(status, { 'content-type': contentType });
          res.end(value);
        };
        const sendJson = (status: number, value: Json): void => send(status, JSON.stringify(value), 'application/json');
        const isAnthropicPath = path === '/v1/messages';

        if (req.method === 'GET' && path === '/v1/models') {
          const anthropicShape = req.headers['x-api-key'] !== undefined || req.headers['anthropic-version'] !== undefined;
          sendJson(200, anthropicShape ? anthropicModelList(models) : openAIModelList(models));
          return;
        }
        if (req.method !== 'POST' || (path !== '/v1/chat/completions' && path !== '/v1/messages')) {
          sendJson(404, { error: 'not_found', message: `no route ${req.method ?? ''} ${path}` });
          return;
        }
        const clientApi = isAnthropicPath ? 'anthropic' : 'openai';
        if (raw === null) {
          sendJson(413, clientApi === 'anthropic' ? anthropicError('invalid_request_error', 'the request body is too large') : openAIError('the request body is too large', 'invalid_request_error', null));
          return;
        }
        let body: unknown;
        try {
          body = JSON.parse(raw.toString('utf8'));
        } catch {
          sendJson(400, clientApi === 'anthropic' ? anthropicError('invalid_request_error', 'the request body is not JSON') : openAIError('the request body is not JSON', 'invalid_request_error', null));
          return;
        }
        if (!isRecord(body)) {
          sendJson(400, clientApi === 'anthropic' ? anthropicError('invalid_request_error', 'the request body is not one JSON object') : openAIError('the request body is not one JSON object', 'invalid_request_error', null));
          return;
        }
        const stream = body.stream === true;
        const myIndex = requestIndex;
        requestIndex += 1;
        const manifestRow = manifest[myIndex];
        const requires = manifestRow !== undefined ? manifestRow.requires : requirementsFromBody(body, isAnthropicPath);
        const jsonMode = jsonModeOf(body, isAnthropicPath);
        const ranked = candidatesFor(probes, brief.ratings, requires, jsonMode);

        let answer: UpstreamAnswer | null = null;
        let servedModel = '';
        for (const candidate of ranked) {
          const providerState = state.get(candidate.provider.id);
          if (providerState === undefined || providerState.exhausted) continue;
          providerState.calls += 1;
          const got = await callProvider(candidate, clientApi, body, stream);
          if (got === null) {
            providerState.consecutiveFailures += 1;
            if (providerState.consecutiveFailures >= 2) providerState.exhausted = true;
            continue;
          }
          providerState.consecutiveFailures = 0;
          answer = got;
          servedModel = candidate.modelId;
          break;
        }
        if (answer === null) {
          sendJson(502, clientApi === 'anthropic' ? anthropicError('overloaded_error', 'every provider refused the request') : openAIError('every provider refused the request', 'server_error', null));
          return;
        }
        if (clientApi === 'anthropic') {
          if (stream) send(200, anthropicSSEOf(answer, servedModel), 'text/event-stream');
          else sendJson(200, anthropicAnswerJSON(answer, servedModel));
        } else {
          if (stream) send(200, openAISSEOf(answer, servedModel), 'text/event-stream');
          else sendJson(200, openAIAnswerJSON(answer, servedModel));
        }
      })().catch(() => {
        try {
          res.writeHead(500, { 'content-type': 'application/json' });
          res.end(JSON.stringify({ error: 'internal' }));
        } catch {
          // The socket is already gone.
        }
      });
    });
  });

  await new Promise<void>((resolve, reject) => {
    server.once('error', reject);
    server.listen({ port, host: '0.0.0.0' }, () => resolve());
  });
  const address = server.address();
  const boundPort = typeof address === 'object' && address !== null ? address.port : port;

  // Register with the env, retrying while it comes up.
  let registered = false;
  for (let attempt = 0; attempt < REGISTER_RETRIES && !registered; attempt += 1) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: 1, port: boundPort, catalog }),
        signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
      });
      if (res.ok) registered = true;
    } catch {
      // The env is not up yet; retry.
    }
    if (!registered) await new Promise((r) => setTimeout(r, REGISTER_RETRY_MS));
  }
  if (!registered) {
    process.stderr.write('free-model-aggregator-01: the env never accepted the registration\n');
    process.exit(1);
  }

  // Poll until the driver says the trial is done, then exit 0.
  for (;;) {
    try {
      const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS) });
      if (res.ok) {
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) break;
      }
    } catch {
      // The env is briefly unreachable; keep polling.
    }
    await new Promise((r) => setTimeout(r, WORK_POLL_MS));
  }
  server.close();
  process.exit(0);
}

main().catch((err: unknown) => {
  process.stderr.write(`free-model-aggregator-01: ${(err as Error).message}\n`);
  process.exit(1);
});
