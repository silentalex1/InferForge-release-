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

  const body = await upstream.text()
  return new Response(body, {
    status: upstream.status >= 500 ? 424 : upstream.status,
    headers: {
      "Content-Type": upstream.headers.get("content-type") || "application/json; charset=utf-8",
      "Cache-Control": "no-store",
    },
  })
}
