import { Copy, Check, Download as DownloadIcon, Terminal, Package, Apple, Monitor, Zap, ArrowRight } from 'lucide-react'
import { useState } from 'react'
import { Link } from 'react-router-dom'

type PlatformId = 'windows' | 'macos' | 'linux' | 'pip'

export default function Download() {
  const [copied, setCopied] = useState<string | null>(null)
  const [activePlatform, setActivePlatform] = useState<PlatformId>('windows')

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text)
    setCopied(id)
    setTimeout(() => setCopied(null), 2000)
  }

  const platforms: {
    id: PlatformId
    label: string
    icon: JSX.Element
    command: string
    description: string
  }[] = [
    {
      id: 'windows',
      label: 'Windows',
      icon: <Monitor className="w-5 h-5" />,
      command: 'powershell -c "irm https://inferforge.org/install.ps1 | iex"',
      description: 'One line in PowerShell. Installs the built forge CLI from the InferForge package index, no source checkout.',
    },
    {
      id: 'macos',
      label: 'macOS',
      icon: <Apple className="w-5 h-5" />,
      command: 'curl -fsSL https://inferforge.org/install.sh | bash',
      description: 'Installs the built forge CLI on Apple Silicon and Intel.',
    },
    {
      id: 'linux',
      label: 'Linux',
      icon: <Terminal className="w-5 h-5" />,
      command: 'curl -fsSL https://inferforge.org/install.sh | bash',
      description: 'Universal installer for Debian, Ubuntu, Fedora, and Arch.',
    },
    {
      id: 'pip',
      label: 'pip',
      icon: <Package className="w-5 h-5" />,
      command: 'pip install inferforge --index-url https://hyperneural.cfd/pypi/simple/ --extra-index-url https://pypi.org/simple',
      description: 'Install directly into any Python 3.10+ environment from the InferForge package index.',
    },
  ]

  const active = platforms.find(p => p.id === activePlatform)!

  const quickstart = [
    { cmd: 'forge pull qwen2.5-coder:7b', desc: 'Download a coding model from HuggingFace' },
    { cmd: 'forge chat', desc: 'Start an interactive chat session' },
    { cmd: 'forge serve', desc: 'Launch an OpenAI-compatible API server' },
  ]

  const releaseNotes = [
    {
      title: 'Plugin System',
      body: 'Extend InferForge with custom commands from a single Python file in ~/.inferforge/plugins.',
    },
    {
      title: 'Configuration Profiles',
      body: 'Create GPU development and CPU production profiles, then switch between them instantly.',
    },
    {
      title: 'Model Versioning & Rollback',
      body: 'Tag training iterations, diff versions side by side, and roll back at any time.',
    },
    {
      title: 'Smart Caching Layer',
      body: 'Automatic prompt and KV caching with hit-rate reporting via forge cache stats.',
    },
    {
      title: 'Quantization Optimizer',
      body: 'Automatically pick the best quantization for your hardware with speed or quality profiles.',
    },
    {
      title: 'Usage Analytics & API Keys',
      body: 'Track token usage and performance metrics, and manage provider keys securely.',
    },
  ]

  return (
    <div className="py-20 px-6 max-w-6xl mx-auto">
      <div className="text-center mb-16">
        <div className="inline-flex p-4 bg-gradient-to-br from-primary/20 to-accent/20 rounded-2xl mb-6 shadow-lg shadow-primary/10">
          <DownloadIcon className="w-10 h-10 text-primary" />
        </div>
        <h1 className="text-4xl md:text-5xl font-bold mb-4 tracking-tight">Install InferForge</h1>
        <p className="text-lg md:text-xl text-gray-400">Up and running in under a minute.</p>
        <a href="/InferForgeInstaller.exe" download className="inline-flex items-center gap-2 mt-6 px-8 py-3.5 rounded-xl bg-[#3b82f6] text-white font-bold hover:bg-[#2563eb] transition">Download the InferForge Installer (.exe) <DownloadIcon className="w-5 h-5" /></a>
        <p className="text-xs text-white/30 mt-2">Guided Windows setup. The one-line command below installs the same build.</p>
      </div>

      <div className="card max-w-3xl mx-auto mb-16 p-0 overflow-hidden">
        <div className="grid grid-cols-2 sm:grid-cols-4 border-b border-white/10 bg-white/[0.02]">
          {platforms.map(p => (
            <button
              key={p.id}
              onClick={() => setActivePlatform(p.id)}
              className={`flex items-center justify-center gap-2 px-4 py-3.5 text-sm font-medium transition-colors ${
                activePlatform === p.id
                  ? 'text-primary border-b-2 border-primary bg-primary/5'
                  : 'text-gray-400 hover:text-white hover:bg-white/[0.03]'
              }`}
            >
              {p.icon}
              {p.label}
            </button>
          ))}
        </div>

        <div className="p-6">
          <p className="text-sm text-gray-400 mb-4">{active.description}</p>
          <div className="flex items-stretch gap-3">
            <code className="flex-1 flex items-center bg-black/60 px-4 py-3.5 rounded-lg font-mono text-sm text-accent border border-white/10 overflow-x-auto whitespace-nowrap">
              {active.command}
            </code>
            <button
              onClick={() => copyToClipboard(active.command, active.id)}
              aria-label="Copy to clipboard"
              className="px-4 rounded-lg border border-white/10 bg-white/5 hover:bg-white/10 transition-colors flex-shrink-0"
            >
              {copied === active.id ? (
                <Check className="w-5 h-5 text-green-400" />
              ) : (
                <Copy className="w-5 h-5 text-gray-300" />
              )}
            </button>
          </div>
        </div>
      </div>

      <div className="mb-16">
        <div className="flex items-center gap-3 mb-6">
          <div className="p-2 bg-primary/10 rounded-lg">
            <Zap className="w-5 h-5 text-primary" />
          </div>
          <h2 className="text-2xl font-bold tracking-tight">Quick Start</h2>
        </div>

        <div className="card p-0 overflow-hidden">
          {quickstart.map((item, index) => (
            <div key={index} className="flex items-start gap-4 px-6 py-4 border-b border-white/5 last:border-0 hover:bg-white/[0.02] transition-colors">
              <span className="flex items-center justify-center w-7 h-7 rounded-full bg-primary/15 text-primary text-xs font-bold flex-shrink-0 mt-0.5">
                {index + 1}
              </span>
              <div className="min-w-0">
                <code className="font-mono text-sm text-accent break-all">{item.cmd}</code>
                <div className="text-gray-500 text-xs mt-1">{item.desc}</div>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="mb-16">
        <h2 className="text-2xl font-bold mb-6 tracking-tight">What is new in v0.3.0</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {releaseNotes.map((note, index) => (
            <div key={index} className="card">
              <div className="flex items-start gap-3">
                <span className="mt-1.5 w-2 h-2 rounded-full bg-gradient-to-br from-primary to-accent flex-shrink-0" />
                <div>
                  <strong className="text-white">{note.title}</strong>
                  <p className="text-gray-400 text-sm mt-1 leading-relaxed">{note.body}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="card bg-gradient-to-br from-primary/10 to-accent/10 border-primary/25 text-center py-12">
        <h3 className="text-2xl font-bold mb-3">Installed? Keep going.</h3>
        <p className="text-gray-400 mb-6">Learn every command and start training your own models.</p>
        <Link to="/docs" className="btn-primary inline-flex items-center gap-2">
          Open Documentation
          <ArrowRight className="w-4 h-4" />
        </Link>
      </div>
    </div>
  )
}
