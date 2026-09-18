import { FormEvent, useEffect, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router-dom'

type Role = 'user' | 'assistant' | 'system'

interface Message {
  id: string
  role: Role
  content: string
}

interface Thread {
  id: string
  title: string
  messages: Message[]
}

const MODELS = ['inferforge-beta']
const STORAGE_KEY = 'inferforge-chat-threads'

function uid() {
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`
}

function loadThreads(): Thread[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return []
    const parsed = JSON.parse(raw)
    return Array.isArray(parsed) ? parsed : []
  } catch {
    return []
  }
}

export default function Chat() {
  const [threads, setThreads] = useState<Thread[]>(() => loadThreads())
  const [activeId, setActiveId] = useState<string | null>(null)
  const [input, setInput] = useState('')
  const [model, setModel] = useState(MODELS[0])
  const [busy, setBusy] = useState(false)
  const [loginOpen, setLoginOpen] = useState(false)
  const [userName, setUserName] = useState(() => {
    try {
      const raw = localStorage.getItem('inferforge-user')
      if (!raw) return ''
      const parsed = JSON.parse(raw)
      return typeof parsed === 'string' ? parsed : parsed.username || parsed.name || raw
    } catch { return localStorage.getItem('inferforge-user') || '' }
  })
  const displayName = (() => {
    try {
      const raw = userName
      if (!raw) return ''
      if (raw.trim().startsWith('{')) {
        const p = JSON.parse(raw)
        return p.username || p.name || ''
      }
      return raw
    } catch { return userName }
  })()
  const bottomRef = useRef<HTMLDivElement>(null)

  const active = useMemo(
    () => threads.find(t => t.id === activeId) ?? null,
    [threads, activeId]
  )

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(threads))
  }, [threads])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [active?.messages, busy])

  const ensureThread = () => {
    if (active) return active
    const created: Thread = { id: uid(), title: 'New chat', messages: [] }
    setThreads(prev => [created, ...prev])
    setActiveId(created.id)
    return created
  }

  const updateThread = (id: string, fn: (t: Thread) => Thread) => {
    setThreads(prev => prev.map(t => (t.id === id ? fn(t) : t)))
  }

  const send = async (event?: FormEvent) => {
    event?.preventDefault()
    const text = input.trim()
    if (!text || busy) return
    const thread = ensureThread()
    const userMsg: Message = { id: uid(), role: 'user', content: text }
    const assistantMsg: Message = { id: uid(), role: 'assistant', content: '' }
    const nextMessages = [...thread.messages, userMsg, assistantMsg]
    updateThread(thread.id, t => ({
      ...t,
      title: t.messages.length === 0 ? text.slice(0, 42) : t.title,
      messages: nextMessages,
    }))
    setInput('')
    setBusy(true)
    try {
      const res = await fetch('https://hyperneural.cfd/api/ai/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model,
          messages: nextMessages
            .filter(m => m.content)
            .map(m => ({ role: m.role, content: m.content })),
        }),
      })
      if (!res.ok || !res.body) {
        const err = await res.text()
        updateThread(thread.id, t => ({
          ...t,
          messages: t.messages.map(m =>
            m.id === assistantMsg.id
              ? { ...m, content: err || 'InferForge could not reply. Try again.' }
              : m
          ),
        }))
        return
      }
      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let acc = ''
      let buffer = ''
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const parts = buffer.split('\n\n')
        buffer = parts.pop() || ''
        for (const part of parts) {
          const line = part.replace(/^data:\s*/, '').trim()
          if (!line || line === '[DONE]') continue
          try {
            const json = JSON.parse(line)
            const piece = json.choices?.[0]?.delta?.content ?? json.choices?.[0]?.message?.content ?? json.content ?? json.response ?? ''
            if (!piece) continue
            acc += piece
            const snapshot = acc
            updateThread(thread.id, t => ({
              ...t,
              messages: t.messages.map(m =>
                m.id === assistantMsg.id ? { ...m, content: snapshot } : m
              ),
            }))
          } catch {
            acc += line
          }
        }
      }
      if (!acc) {
        updateThread(thread.id, t => ({
          ...t,
          messages: t.messages.map(m =>
            m.id === assistantMsg.id
              ? { ...m, content: 'Ready when you are. Ask InferForge anything.' }
              : m
          ),
        }))
      }
    } catch {
      updateThread(thread.id, t => ({
        ...t,
        messages: t.messages.map(m =>
          m.id === assistantMsg.id
            ? { ...m, content: 'Network error. Check your connection and retry.' }
            : m
        ),
      }))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="h-screen w-screen bg-white text-[#111] flex overflow-hidden">
      <aside className="w-[240px] shrink-0 bg-[#f6f7f9] border-r border-[#ececec] flex flex-col">
        <div className="px-5 py-4 text-[15px] font-medium text-[#222]">Chat History</div>
        <div className="flex-1 overflow-y-auto px-2 pb-4">
          {threads.length === 0 && (
            <p className="px-3 py-2 text-sm text-[#9aa0a6]">No chats yet</p>
          )}
          {threads.map(thread => (
            <button
              key={thread.id}
              onClick={() => setActiveId(thread.id)}
              className={`w-full text-left px-3 py-2 rounded-lg text-sm truncate ${
                thread.id === activeId ? 'bg-white shadow-sm' : 'hover:bg-white/70'
              }`}
            >
              {thread.title}
            </button>
          ))}
        </div>
        <button
          onClick={() => {
            const created: Thread = { id: uid(), title: 'New chat', messages: [] }
            setThreads(prev => [created, ...prev])
            setActiveId(created.id)
          }}
          className="m-3 text-sm text-[#555] hover:text-black"
        >
          New chat
        </button>
      </aside>

      <div className="flex-1 flex flex-col min-w-0">
        <header className="h-14 border-b border-[#ececec] flex items-center justify-between px-5">
          <div className="flex items-center gap-2">
            <Link to="/" className="font-semibold text-[17px] tracking-tight">InferForge</Link>
            <span className="text-[11px] px-2 py-0.5 rounded-full bg-[#f1f2f4] text-[#666]">stable release</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-sm text-emerald-600 font-medium">Welcome {displayName || 'guest'}</span>
            <select
              value={model}
              onChange={e => setModel(e.target.value)}
              className="h-9 rounded-full border border-[#e5e7eb] px-3 text-sm bg-white"
            >
              {MODELS.map(m => (
                <option key={m} value={m}>{m}</option>
              ))}
            </select>
          </div>
        </header>

        <div className="flex-1 overflow-y-auto px-6">
          {(!active || active.messages.length === 0) && (
            <div className="h-full" />
          )}
          {active?.messages.map(msg => (
            <div key={msg.id} className="max-w-3xl mx-auto py-4">
              <div className="text-xs text-[#888] mb-1">{msg.role === 'user' ? (displayName || 'You') : 'InferForge'}</div>
              <div className="whitespace-pre-wrap leading-7 text-[15px]">{msg.content}</div>
            </div>
          ))}
          <div ref={bottomRef} />
        </div>

        <div className="px-6 pb-6">
          <form onSubmit={send} className="max-w-2xl mx-auto">
            <div className="flex items-center gap-2 rounded-full border border-[#e5e7eb] bg-white shadow-[0_8px_30px_rgba(0,0,0,0.06)] px-5 py-2">
              <input
                value={input}
                onChange={e => setInput(e.target.value)}
                placeholder="Ask InferForge anything.."
                className="flex-1 h-10 outline-none text-[15px] bg-transparent"
              />
              <button
                type="submit"
                disabled={busy}
                className="w-9 h-9 rounded-full bg-black text-white flex items-center justify-center disabled:opacity-40"
                aria-label="Send"
              >
                ▶
              </button>
            </div>
            <p className="text-center text-xs text-[#9aa0a6] mt-3">Ready · {model}</p>
          </form>
        </div>
      </div>

      {loginOpen && (
        <div className="fixed inset-0 bg-black/30 flex items-center justify-center z-30">
          <form
            className="bg-white rounded-2xl p-6 w-[320px] shadow-xl"
            onSubmit={e => {
              e.preventDefault()
              try {
                const trimmed = userName.trim()
                const obj = trimmed.startsWith('{') ? JSON.parse(trimmed) : { username: trimmed }
                const name = obj.username || trimmed
                localStorage.setItem('inferforge-user', JSON.stringify({ username: name }))
              } catch {
                localStorage.setItem('inferforge-user', JSON.stringify({ username: userName }))
              }
              setLoginOpen(false)
              window.location.reload()
            }}
          >
            <h2 className="font-semibold mb-3">Login</h2>
            <input
              value={userName}
              onChange={e => setUserName(e.target.value)}
              placeholder="Display name"
              className="w-full border border-[#e5e7eb] rounded-lg px-3 py-2 mb-4"
            />
            <div className="flex justify-end gap-2">
              <button type="button" onClick={() => setLoginOpen(false)} className="text-sm text-[#666]">Cancel</button>
              <button type="submit" className="text-sm px-3 py-1.5 rounded-lg bg-black text-white">Save</button>
            </div>
          </form>
        </div>
      )}
    </div>
  )
}
