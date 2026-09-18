export interface Env {
  FROM_EMAIL: string
  FROM_NAME: string
  RESEND_API_KEY: string
  CONNECT_KV: KVNamespace
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const cors = {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type",
    }
    const url = new URL(request.url)

    if (request.method === "OPTIONS") return new Response(null, { headers: cors })

    if (url.pathname.startsWith("/api/auth/") && request.method === "OPTIONS") return new Response(null, { headers: cors })

    if (url.pathname === "/api/auth/register" && request.method === "POST") {
      try {
        const body = await request.json() as { username?: string; email?: string; password?: string }
        const username = String(body.username || "").trim().toLowerCase()
        const email = String(body.email || "").trim().toLowerCase()
        const password = String(body.password || "")
        if (!username || !/^[a-z0-9_]{3,24}$/.test(username)) return new Response(JSON.stringify({ error: "username must be 3-24 chars (a-z, 0-9, _)" }), { status: 400, headers: { "Content-Type": "application/json", ...cors } })
        if (!email || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) return new Response(JSON.stringify({ error: "valid email required" }), { status: 400, headers: { "Content-Type": "application/json", ...cors } })
        if (password.length < 6) return new Response(JSON.stringify({ error: "password must be at least 6 characters" }), { status: 400, headers: { "Content-Type": "application/json", ...cors } })
        const exists = await env.CONNECT_KV.get(`user:${username}`)
        if (exists) return new Response(JSON.stringify({ error: "username already taken" }), { status: 409, headers: { "Content-Type": "application/json", ...cors } })
        const hashBuf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(`${username}:${password}`))
        const pwhash = Array.from(new Uint8Array(hashBuf)).map(b => b.toString(16).padStart(2, "0")).join("")
        await env.CONNECT_KV.put(`user:${username}`, JSON.stringify({ username, email, pwhash, created: new Date().toISOString() }))
        const idx = await env.CONNECT_KV.get("users:index")
        const users: string[] = idx ? JSON.parse(idx) : []
        if (!users.includes(username)) { users.push(username); await env.CONNECT_KV.put("users:index", JSON.stringify(users)) }
        return new Response(JSON.stringify({ ok: true, user: { username, email } }), { headers: { "Content-Type": "application/json", ...cors } })
      } catch {
        return new Response(JSON.stringify({ error: "invalid request" }), { status: 400, headers: { "Content-Type": "application/json", ...cors } })
      }
    }

    if (url.pathname === "/api/auth/login" && request.method === "POST") {
      try {
        const body = await request.json() as { username?: string; password?: string }
        const username = String(body.username || "").trim().toLowerCase()
        const password = String(body.password || "")
        const raw = await env.CONNECT_KV.get(`user:${username}`)
        if (!raw) return new Response(JSON.stringify({ error: "account not found" }), { status: 404, headers: { "Content-Type": "application/json", ...cors } })
        const acct = JSON.parse(raw) as { username: string; email: string; pwhash: string }
        const hashBuf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(`${username}:${password}`))
        const pwhash = Array.from(new Uint8Array(hashBuf)).map(b => b.toString(16).padStart(2, "0")).join("")
        if (pwhash !== acct.pwhash) return new Response(JSON.stringify({ error: "incorrect password" }), { status: 403, headers: { "Content-Type": "application/json", ...cors } })
        return new Response(JSON.stringify({ ok: true, user: { username: acct.username, email: acct.email } }), { headers: { "Content-Type": "application/json", ...cors } })
      } catch {
        return new Response(JSON.stringify({ error: "invalid request" }), { status: 400, headers: { "Content-Type": "application/json", ...cors } })
      }
    }

    if (url.pathname === "/api/auth/reset-request" && request.method === "POST") {
      try {
        const body = await request.json() as { username?: string; code?: string }
        const username = String(body.username || "").trim().toLowerCase()
        const code = String(body.code || "").trim().toUpperCase()
        if (!username || !/^[A-Z0-9]{5,8}$/.test(code)) return new Response(JSON.stringify({ error: "username and code required" }), { status: 400, headers: { "Content-Type": "application/json", ...cors } })
        const raw = await env.CONNECT_KV.get(`user:${username}`)
        if (!raw) return new Response(JSON.stringify({ error: "account not found" }), { status: 404, headers: { "Content-Type": "application/json", ...cors } })
        await env.CONNECT_KV.put(`reset:${username}`, JSON.stringify({ code, issued: new Date().toISOString() }), { expirationTtl: 600 })
        return new Response(JSON.stringify({ ok: true, username, expires_in: 600 }), { headers: { "Content-Type": "application/json", ...cors } })
      } catch {
        return new Response(JSON.stringify({ error: "invalid request" }), { status: 400, headers: { "Content-Type": "application/json", ...cors } })
      }
    }

    if (url.pathname === "/api/auth/reset" && request.method === "POST") {
      try {
        const body = await request.json() as { username?: string; password?: string; code?: string }
        const username = String(body.username || "").trim().toLowerCase()
        const password = String(body.password || "")
        const code = String(body.code || "").trim().toUpperCase()
        if (!username) return new Response(JSON.stringify({ error: "username required" }), { status: 400, headers: { "Content-Type": "application/json", ...cors } })
        if (password.length < 6) return new Response(JSON.stringify({ error: "password must be at least 6 characters" }), { status: 400, headers: { "Content-Type": "application/json", ...cors } })
        const raw = await env.CONNECT_KV.get(`user:${username}`)
        if (!raw) return new Response(JSON.stringify({ error: "account not found" }), { status: 404, headers: { "Content-Type": "application/json", ...cors } })
        const pending = await env.CONNECT_KV.get(`reset:${username}`)
        const issued = pending ? (JSON.parse(pending) as { code?: string }).code || "" : ""
        if (!code || !issued || code !== issued) return new Response(JSON.stringify({ error: "reset code missing or expired. Run: forge account reset" }), { status: 403, headers: { "Content-Type": "application/json", ...cors } })
        const acct = JSON.parse(raw) as { username: string; email: string; pwhash: string; created?: string }
        const hashBuf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(`${username}:${password}`))
        const pwhash = Array.from(new Uint8Array(hashBuf)).map(b => b.toString(16).padStart(2, "0")).join("")
        await env.CONNECT_KV.put(`user:${username}`, JSON.stringify({ ...acct, pwhash, reset_at: new Date().toISOString() }))
        await env.CONNECT_KV.delete(`reset:${username}`)
        return new Response(JSON.stringify({ ok: true, user: { username: acct.username, email: acct.email } }), { headers: { "Content-Type": "application/json", ...cors } })
      } catch {
        return new Response(JSON.stringify({ error: "invalid request" }), { status: 400, headers: { "Content-Type": "application/json", ...cors } })
      }
    }

    if (url.pathname.startsWith("/api/auth/user/") && request.method === "GET") {
      const username = url.pathname.split("/").pop()?.toLowerCase()
      if (!username) return new Response(JSON.stringify({ error: "username required" }), { status: 400, headers: { "Content-Type": "application/json", ...cors } })
      const raw = await env.CONNECT_KV.get(`user:${username}`)
      if (!raw) return new Response(JSON.stringify({ error: "not found" }), { status: 404, headers: { "Content-Type": "application/json", ...cors } })
      const acct = JSON.parse(raw) as { username: string; email: string; created?: string }
      return new Response(JSON.stringify({ username: acct.username, email: acct.email, created: acct.created || null }), { headers: { "Content-Type": "application/json", ...cors } })
    }

    if (url.pathname === "/api/auth/users-count" && request.method === "GET") {
      const idx = await env.CONNECT_KV.get("users:index")
      const users: string[] = idx ? JSON.parse(idx) : []
      return new Response(JSON.stringify({ count: users.length }), { headers: { "Content-Type": "application/json", ...cors } })
    }

    if (url.pathname === "/connect" && request.method === "POST") {
      try {
        const body = await request.json() as { username?: string; code?: string; email?: string; confirm?: boolean }
        const username = String(body.username || "").trim().toLowerCase()
        if (!username) return new Response(JSON.stringify({ error: "username required" }), { status: 400, headers: { "Content-Type": "application/json", ...cors } })
        const key = `connect:${username}`
        if (body.confirm) {
          const existing = await env.CONNECT_KV.get(key)
          if (!existing) return new Response(JSON.stringify({ error: "no pending session" }), { status: 404, headers: { "Content-Type": "application/json", ...cors } })
          const session = JSON.parse(existing) as { code: string }
          if (String(body.code || "").toUpperCase() !== session.code) return new Response(JSON.stringify({ error: "code mismatch" }), { status: 403, headers: { "Content-Type": "application/json", ...cors } })
          await env.CONNECT_KV.put(key, JSON.stringify({ code: session.code, email: String(body.email || ""), confirmed: true }), { expirationTtl: 600 })
          return new Response(JSON.stringify({ ok: true, confirmed: true }), { headers: { "Content-Type": "application/json", ...cors } })
        }
        const code = String(body.code || "").toUpperCase()
        if (!code) return new Response(JSON.stringify({ error: "code required" }), { status: 400, headers: { "Content-Type": "application/json", ...cors } })
        await env.CONNECT_KV.put(key, JSON.stringify({ code, confirmed: false }), { expirationTtl: 600 })
        return new Response(JSON.stringify({ ok: true }), { headers: { "Content-Type": "application/json", ...cors } })
      } catch (e) {
        return new Response(JSON.stringify({ error: "invalid request" }), { status: 400, headers: { "Content-Type": "application/json", ...cors } })
      }
    }

    if (url.pathname.startsWith("/connect/") && request.method === "GET") {
      const username = url.pathname.split("/")[2]?.toLowerCase()
      if (!username) return new Response(JSON.stringify({ error: "username required" }), { status: 400, headers: { "Content-Type": "application/json", ...cors } })
      const raw = await env.CONNECT_KV.get(`connect:${username}`)
      if (!raw) return new Response(JSON.stringify({ confirmed: false }), { headers: { "Content-Type": "application/json", ...cors } })
      return new Response(raw, { headers: { "Content-Type": "application/json", ...cors } })
    }

    if (request.method !== "POST") return new Response(JSON.stringify({ error: "Method not allowed" }), { status: 405, headers: { "Content-Type": "application/json", ...cors } })

    let body: { email?: string; code?: string } = {}
    try { body = await request.json() } catch { return new Response(JSON.stringify({ error: "Invalid JSON" }), { status: 400, headers: { "Content-Type": "application/json", ...cors } }) }

    const email = String(body.email || "").trim().toLowerCase()
    const code = String(body.code || "").trim()
    if (!email || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) return new Response(JSON.stringify({ error: "Valid email required" }), { status: 400, headers: { "Content-Type": "application/json", ...cors } })
    if (!code) return new Response(JSON.stringify({ error: "Code required" }), { status: 400, headers: { "Content-Type": "application/json", ...cors } })

    const html = `<div style="font-family:Inter,sans-serif;max-width:480px;margin:0 auto;padding:28px 24px;background:#0A0A0B;color:#fff;border-radius:16px;border:1px solid rgba(255,255,255,0.07);"><div style="text-align:center;margin-bottom:20px;"><span style="width:28px;height:28px;border-radius:8px;background:#fff;color:#000;display:inline-flex;align-items:center;justify-content:center;font-size:11px;font-weight:800;">IF</span><span style="margin-left:8px;font-size:14px;font-weight:700;letter-spacing:-0.02em;">InferForge</span></div><h2 style="margin:0 0 8px;font-size:20px;font-weight:700;letter-spacing:-0.02em;text-align:center;">Your verification code</h2><p style="margin:0;color:rgba(255,255,255,0.55);font-size:13px;line-height:20px;text-align:center;">Use this code to verify your email. It expires in 5 minutes.</p><div style="margin:22px 0;padding:18px;text-align:center;background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.08);border-radius:12px;font-size:28px;letter-spacing:8px;font-weight:800;">${code}</div><p style="margin:0;font-size:12px;color:rgba(255,255,255,0.35);text-align:center;line-height:18px;">If you did not request this, you can ignore this email.</p></div>`

    if (env.RESEND_API_KEY) {
      try {
        const r = await fetch("https://api.resend.com/emails", {
          method: "POST",
          headers: { Authorization: `Bearer ${env.RESEND_API_KEY}`, "Content-Type": "application/json" },
          body: JSON.stringify({
            from: "InferForge <onboarding@resend.dev>",
            to: [email],
            subject: `Your InferForge code is ${code}`,
            html,
            text: `Your InferForge verification code is ${code}. It expires in 5 minutes.`,
          }),
        })
        if (r.ok) return new Response(JSON.stringify({ ok: true }), { headers: { "Content-Type": "application/json", ...cors } })
      } catch {}
    }

    try {
      const res = await fetch("https://api.mailchannels.net/tx/v1/send", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          personalizations: [{ to: [{ email, name: email }] }],
          from: { email: env.FROM_EMAIL || "noreply@hyperneural.cfd", name: env.FROM_NAME || "InferForge" },
          subject: `Your InferForge code is ${code}`,
          content: [
            { type: "text/plain", value: `Your InferForge verification code is ${code}. It expires in 5 minutes.` },
            { type: "text/html", value: html },
          ],
        }),
      })
      if (!res.ok) {
        const txt = await res.text()
        console.log(`MailChannels failed for ${email}: ${txt}`)
      }
    } catch (e) {
      console.log(`MailChannels error for ${email}: ${e}`)
    }
    return new Response(JSON.stringify({ ok: true }), { headers: { "Content-Type": "application/json", ...cors } })
  },
}