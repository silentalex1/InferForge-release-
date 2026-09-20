import {
  cleanEndpoint,
  corsHeaders,
  deleteRecord,
  deleteStats,
  hashToken,
  HOSTED_MODEL,
  HOSTED_MODELS,
  addOwnerSlug,
  isHttpUrl,
  isPremium,
  json,
  ownerSlugs,
  removeOwnerSlug,
  publicView,
  readRecord,
  slugify,
  timingSafeEqual,
  writeRecord,
} from "../../../lib/sdk/store"
import type { SdkRecord } from "../../../lib/sdk/store"

const MAX_MODEL_LENGTH = 96

const FREE_MODEL_LIMIT = 2

function token(request: Request): string {
  return (request.headers.get("x-forge-token") || "").trim()
}

export async function onRequest(context: any): Promise<Response> {
  const request: Request = context.request
  const kv = context.env.INFERFORGE_SDK

  if (request.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders() })
  }
  if (!kv) {
    return json({ error: "registry-unavailable", message: "The SDK registry is not configured." }, 503)
  }

  if (request.method === "GET") {
    const slug = slugify(new URL(request.url).searchParams.get("model") || "")
    const record = await readRecord(kv, slug)
    if (!record) return json({ error: "not-found" }, 404)
    return json(publicView(record))
  }

  const presented = token(request)
  if (!presented) {
    return json({ error: "missing-token", message: "Send your publish token in the X-Forge-Token header." }, 401)
  }
  const presentedHash = await hashToken(presented)

  if (request.method === "DELETE") {
    let body: any = {}
    try {
      body = await request.json()
    } catch {
      body = {}
    }
    const slug = slugify(body?.slug || body?.model || "")
    const record = await readRecord(kv, slug)
    if (!record) return json({ error: "not-found" }, 404)
    if (!timingSafeEqual(presentedHash, record.token_hash)) {
      return json({ error: "forbidden", message: "This publish token does not own '" + slug + "'." }, 403)
    }
    await deleteRecord(kv, slug)
    await deleteStats(kv, slug)
    await removeOwnerSlug(kv, record.owner, slug)
    return json({ ok: true, slug })
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
  if (!model || model.length > MAX_MODEL_LENGTH) {
    return json({ error: "invalid-model", message: "Provide a model name of 1-96 characters." }, 400)
  }

  const slug = slugify(body?.slug || model)
  const key = String(body?.key || "").trim()
  if (!key.startsWith("sk-embed-")) {
    return json({ error: "invalid-key", message: "The embed key must start with sk-embed-." }, 400)
  }

  const endpoint = cleanEndpoint(body?.endpoint || "")
  if (!isHttpUrl(endpoint)) {
    return json({ error: "invalid-endpoint", message: "Provide an http(s) endpoint for the model server." }, 400)
  }
  const fallback = cleanEndpoint(body?.fallback || "")
  if (fallback && !isHttpUrl(fallback)) {
    return json({ error: "invalid-fallback", message: "The fallback endpoint must be an http(s) URL." }, 400)
  }

  const system = String(body?.system || "").slice(0, 8000)
  const requestedHosted = String(body?.hosted_model || "").trim()
  const hosted_model = HOSTED_MODELS.includes(requestedHosted) ? requestedHosted : HOSTED_MODEL

  const existing = await readRecord(kv, slug)
  if (existing && !timingSafeEqual(presentedHash, existing.token_hash)) {
    return json(
      {
        error: "name-taken",
        message: "'" + slug + "' is published by another account. Rename the model or unpublish it first.",
      },
      403
    )
  }

  const owner = String(body?.owner || "").trim().slice(0, 64)

  if (!existing && owner && !(await isPremium(kv, owner))) {
    const mine = await ownerSlugs(kv, owner)
    if (!mine.includes(slug) && mine.length >= FREE_MODEL_LIMIT) {
      return json(
        {
          error: "model-limit-reached",
          limit: FREE_MODEL_LIMIT,
          hosted: mine.length,
          message:
            "The free plan hosts up to " + FREE_MODEL_LIMIT + " AI models and you already have " + mine.length +
            " (" + mine.join(", ") + "). Unpublish one with 'forge embedd <model> --sdk --unpublish', " +
            "or upgrade at https://inferforge.org/pricing for unlimited models.",
        },
        403
      )
    }
  }

  const now = new Date().toISOString()
  const record: SdkRecord = {
    model,
    slug,
    key,
    endpoint,
    fallback,
    owner,
    token_hash: presentedHash,
    created_at: existing?.created_at || now,
    updated_at: now,
    system,
    hosted_model,
  }
  await writeRecord(kv, record)
  if (owner) await addOwnerSlug(kv, owner, slug)

  const origin = new URL(request.url).origin
  return json(
    {
      ok: true,
      ...publicView(record),
      sdk_url: origin + "/sdk/" + slug + ".js",
      hosted_model,
      persona_chars: system.length,
      embed: '<script src="' + origin + "/sdk/" + slug + ".js?key=" + key + '&mount=%23inferforge-chat"></script>',
    },
    existing ? 200 : 201
  )
}
