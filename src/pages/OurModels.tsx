import { Link, useNavigate } from 'react-router-dom'
import { Search } from 'lucide-react'
import { useState } from 'react'

const MODELS = [
  { name: 'inferforge-beta', family: 'inferforge', params: '14.8B', desc: 'Flagship local model for chat, code and reasoning.' },
  { name: 'qwen2.5-coder:7b', family: 'qwen', params: '7B', desc: 'Code generation and review.' },
  { name: 'llama3.1:8b', family: 'llama', params: '8B', desc: 'General-purpose instruction model.' },
  { name: 'mistral:7b', family: 'mistral', params: '7B', desc: 'Fast chat and summarization.' },
  { name: 'gemma2:9b', family: 'gemma', params: '9B', desc: 'Research-grade reasoning.' },
  { name: 'fused-coder', family: 'merged', params: '8B', desc: 'Example merged model (TIES).' },
]

export default function OurModels() {
  const [q, setQ] = useState('')
  const navigate = useNavigate()
  const filtered = MODELS.filter(m => `${m.name} ${m.family} ${m.desc}`.toLowerCase().includes(q.toLowerCase()))
  return (
    <div className="min-h-screen bg-[#0A0A0B] text-white">
      <nav className="sticky top-0 z-10 bg-[#0A0A0B]/80 backdrop-blur border-b border-white/[0.06] px-6 py-3 flex items-center gap-4">
        <button onClick={() => navigate(-1)} className="text-sm text-white/60 hover:text-white">&lt; Go back</button>
        <div className="flex-1" />
        <Link to="/docs" className="text-sm text-white/60 hover:text-white">documentation</Link>
        <a href="/#features" className="text-sm text-white/60 hover:text-white">features</a>
      </nav>
      <div className="max-w-4xl mx-auto px-6 py-10">
        <h1 className="text-3xl font-bold">Our current models</h1>
        <div className="relative mt-6 max-w-lg">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-white/30" />
          <input value={q} onChange={e => setQ(e.target.value)} placeholder="Search models..." className="w-full pl-10 pr-4 py-3 rounded-xl bg-white/[0.06] border border-white/[0.08] text-sm text-white placeholder:text-white/30 focus:outline-none focus:border-white/15" />
        </div>
        <div className="grid sm:grid-cols-2 gap-4 mt-6">
          {filtered.map(m => (
            <div key={m.name} className="rounded-2xl border border-white/[0.07] bg-white/[0.03] p-5">
              <div className="flex items-center justify-between">
                <p className="font-mono text-sm font-semibold text-white">{m.name}</p>
                <span className="text-xs px-2 py-1 rounded-full bg-white/[0.06] border border-white/[0.08] text-white/50">{m.params}</span>
              </div>
              <p className="text-xs text-white/40 mt-1">{m.family}</p>
              <p className="text-sm text-white/60 mt-2">{m.desc}</p>
              {m.name === 'inferforge-beta' ? (
                <a href="https://api.inferforge.org/Inferforge:latest#release" target="_blank" rel="noopener" className="text-xs text-emerald-400 hover:underline mt-3 inline-block">SDK: api.inferforge.org/Inferforge:latest#release</a>
              ) : (
                <a href={`https://hyperneural.cfd/${m.name}/#@<username>`} target="_blank" rel="noopener" className="text-xs text-emerald-400 hover:underline mt-3 inline-block">SDK: hyperneural.cfd/{m.name}/#@&lt;username&gt;</a>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
