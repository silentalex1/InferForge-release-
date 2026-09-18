import { Check, Copy } from 'lucide-react'
import { useState } from 'react'

interface Props {
  command: string
}

export default function CopyableCommand({ command }: Props) {
  const [copied, setCopied] = useState(false)
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(command)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch {
      const el = document.createElement('textarea')
      el.value = command
      document.body.appendChild(el)
      el.select()
      try { document.execCommand('copy'); setCopied(true); setTimeout(() => setCopied(false), 2000) } catch {}
      el.remove()
    }
  }
  return (
    <button onClick={copy} aria-label="Copy command" className="p-2 text-gray-500 hover:text-white hover:bg-white/10 rounded-lg transition-colors">
      {copied ? <Check className="w-4 h-4 text-green-400" /> : <Copy className="w-4 h-4" />}
    </button>
  )
}
