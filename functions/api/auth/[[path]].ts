import { createSession } from "../../../lib/sdk/store"

const UPSTREAM = "https://inferforge-email.asdwwas233.workers.dev"

export async function onRequest(context: any): Promise<Response> {
  const request: Request = context.request
  if (request.method === "OPTIONS") {
    return new Response(null, { status: 204 })
  }

  const url = new URL(request.url)
  const segments = (context.params?.path || []) as string[]
  const target = UPSTREAM + "/api/auth/" + segments.map(encodeURIComponent).join("/") + url.search

  const init: RequestInit = {
    method: request.method,
    headers: { "Content-Type": request.headers.get("content-type") || "application/json" },
  }
  if (request.method !== "GET" && request.method !== "HEAD") {
    init.body = await request.text()
  }

  let upstream: Response
  try {
    upstream = await fetch(target, init)
  } catch {
    return new Response(JSON.stringify({ error: "The account service is not responding. Please try again in a moment." }), {
      status: 424,
      headers: { "Content-Type": "application/json; charset=utf-8", "Cache-Control": "no-store" },
    })
  }

  let body = await upstream.text()

  if (upstream.ok && segments[0] === "login") {
    try {
      const parsed = JSON.parse(body)
      if (parsed?.ok && parsed?.user?.username) {
        const session = await createSession(context.env.INFERFORGE_SDK, parsed.user.username)
        if (session) body = JSON.stringify({ ...parsed, session })
      }
    } catch {}
  }

  return new Response(body, {
    status: upstream.status >= 500 ? 424 : upstream.status,
    headers: {
      "Content-Type": upstream.headers.get("content-type") || "application/json; charset=utf-8",
      "Cache-Control": "no-store",
    },
  })
}
