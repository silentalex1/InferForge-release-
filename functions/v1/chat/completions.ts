import {
  bearer,
  bumpStats,
  cleanEndpoint,
  corsHeaders,
  hostedChat,
  isHttpUrl,
  isPrivateHost,
  json,
  probeUpstream,
  readRecord,
  slugify,
  timingSafeEqual,
} from "../../../lib/sdk/store"

const UPSTREAM_FAILED = 424

export async function onRequest(context: any): Promise<Response> {
  const request: Request = context.request

  if (request.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders() })
  }
  if (request.method !== "POST") {
    return json({ error: "method-not-allowed" }, 405)
  }

  let body: any
  try {
    body = await request.json()
  } catch {
    return json({ error: "invalid-json" }, 400)
  }

  const model = String(body?.model || "").trim()
  if (!model) {
    return json({ error: "missing-model", message: "Include a 'model' field." }, 400)
  }

  const record = await readRecord(context.env.INFERFORGE_SDK, slugify(model))
  if (!record) {
    return json(
      { error: "model-not-published", message: "No SDK published for '" + model + "'. Run: forge embedd " + model + " --sdk" },
      404
    )
  }

  if (!timingSafeEqual(bearer(request), record.key)) {
    return json({ error: "invalid-key", message: "The embed key does not match this model." }, 401)
  }

  const slug = record.slug || slugify(model)
  const track = (status: number) => {
    if (typeof context.waitUntil === "function") {
      context.waitUntil(bumpStats(context.env.INFERFORGE_SDK, slug, status))
    }
  }

  const serveHosted = async (): Promise<Response> => {
    const ai = context.env.AI
    if (!ai) {
      track(502)
      return json({ error: "no-hosted-runtime", message: "Hosted inference is not configured for this site." }, UPSTREAM_FAILED)
    }
    try {
      const res = await hostedChat(ai, record, body, { "X-InferForge-Source": "hosted" })
      track(200)
      return res
    } catch (err: any) {
      track(502)
      return json({ error: "hosted-error", message: String(err?.message || "hosted inference failed") }, UPSTREAM_FAILED)
    }
  }

  const upstream = cleanEndpoint(record.endpoint)
  if (!isHttpUrl(upstream) || isPrivateHost(upstream)) {
    return serveHosted()
  }

  const alive = await probeUpstream(upstream, 3000)
  if (!alive) {
    return serveHosted()
  }

  let own: Response
  try {
    own = await fetch(upstream + "/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: "Bearer " + record.key,
        Accept: body?.stream ? "text/event-stream" : "application/json",
      },
      body: JSON.stringify({ ...body, model: record.model || model }),
    })
  } catch {
    return serveHosted()
  }

  if (own.status >= 500) {
    return serveHosted()
  }

  track(own.status)
  return new Response(own.body, {
    status: own.status,
    headers: corsHeaders({
      "Content-Type": own.headers.get("content-type") || "application/json; charset=utf-8",
      "Cache-Control": "no-store",
      "X-InferForge-Source": "own",
    }),
  })
}
