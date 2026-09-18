import {
  bearer,
  cleanEndpoint,
  corsHeaders,
  isHttpUrl,
  json,
  readRecord,
  slugify,
  timingSafeEqual,
} from "../../../lib/sdk/store"

function isPrivateHost(endpoint: string): boolean {
  let host = ""
  try {
    host = new URL(endpoint).hostname.toLowerCase().replace(/^\[|\]$/g, "")
  } catch {
    return true
  }
  if (host === "localhost" || host === "0.0.0.0" || host === "::1" || host.endsWith(".local")) return true
  if (/^127\./.test(host)) return true
  if (/^10\./.test(host)) return true
  if (/^192\.168\./.test(host)) return true
  if (/^169\.254\./.test(host)) return true
  if (/^172\.(1[6-9]|2[0-9]|3[01])\./.test(host)) return true
  return false
}

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
      {
        error: "model-not-published",
        message: "No SDK published for '" + model + "'. Run: forge embedd " + model + " --sdk",
      },
      404
    )
  }

  if (!timingSafeEqual(bearer(request), record.key)) {
    return json({ error: "invalid-key", message: "The embed key does not match this model." }, 401)
  }

  const upstream = cleanEndpoint(record.endpoint)
  if (!isHttpUrl(upstream)) {
    return json({ error: "no-upstream", message: "This model has no reachable server registered." }, 502)
  }
  if (isPrivateHost(upstream)) {
    return json(
      {
        error: "upstream-local",
        message:
          "'" + model + "' is registered on a private address (" + upstream + ") that this site cannot reach. " +
          "Expose `forge serve` on a public HTTPS URL and re-run: forge embedd " + model + " --sdk --endpoint <public-url>",
      },
      502
    )
  }

  let upstreamResponse: Response
  try {
    upstreamResponse = await fetch(upstream + "/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: "Bearer " + record.key,
        Accept: body?.stream ? "text/event-stream" : "application/json",
      },
      body: JSON.stringify({ ...body, model: record.model || model }),
    })
  } catch (err: any) {
    return json(
      {
        error: "upstream-unreachable",
        message: "Could not reach " + upstream + ": " + (err?.message || "connection failed"),
      },
      502
    )
  }

  const headers = corsHeaders({
    "Content-Type": upstreamResponse.headers.get("content-type") || "application/json; charset=utf-8",
    "Cache-Control": "no-store",
  })
  return new Response(upstreamResponse.body, { status: upstreamResponse.status, headers })
}
