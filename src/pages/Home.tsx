import { Link } from 'react-router-dom'
import {
  ArrowRight,
  Bot,
  Check,
  Code2,
  Cpu,
  GitMerge,
  Layers,
  Play,
  Server,
  Shield,
  Terminal,
  Zap,
} from 'lucide-react'
import CopyableCommand from '../components/CopyableCommand'

type Capability = {
  icon: typeof Cpu
  command: string
  title: string
  body: string
}

type Pillar = {
  label: string
  title: string
  body: string
  points: string[]
}

type Tier = {
  name: string
  dot: string
  ring: string
  items: string
}

const capabilities: Capability[] = [
  {
    icon: Play,
    command: 'forge run <model>',
    title: 'Run models',
    body: 'Execute and chat with AI models locally. One router picks the right backend per model: Ollama tags, native GGUF, HuggingFace directories, or a remote endpoint.',
  },
  {
    icon: Cpu,
    command: 'forge train',
    title: 'Train models',
    body: 'Real weight training through the integrated core: LoRA and QLoRA fine-tuning, DPO alignment, agent tool-call training, and from-scratch runs.',
  },
  {
    icon: Server,
    command: 'forge embedd --sdk',
    title: 'Embed & publish',
    body: 'Publish a model and get a hosted SDK at inferforge.org/sdk/<model>.js. Drop one script tag into any site and your local model answers it. Free.',
  },
  {
    icon: GitMerge,
    command: 'forge merge',
    title: 'Merge models',
    body: 'Combine checkpoints with SLERP, TIES, DARE or plain averaging at the GGUF level, and push past what either baseline model could do alone.',
  },
  {
    icon: Code2,
    command: 'forge nexara',
    title: 'Nexara language',
    body: 'A declarative DSL and compiler built for training. Describe the model, the data and the hardware; Nexara compiles it into a tuned training run.',
  },
  {
    icon: Bot,
    command: 'forge test --agent',
    title: 'Agents that act',
    body: 'Models call real tools (read files, edit them, run commands, hit the web) inside a sandbox with workspace limits, consent prompts and an audit log.',
  },
]

const pillars: Pillar[] = [
  {
    label: '01',
    title: 'The runtime',
    body: 'An execution router resolves every model to the backend that can actually serve it, and a registry keeps track of what lives on your machine.',
    points: [
      'Ollama tags, native GGUF via llama.cpp, HuggingFace directories, remote endpoints',
      'Registry tracks metadata, digests, paths and capabilities per model',
      'forge serve exposes an OpenAI-compatible API on :11435 with SSE streaming',
      'API-key auth, rate limiting and CORS built into the server',
    ],
  },
  {
    label: '02',
    title: 'Training',
    body: 'The deepest subsystem. Supervised fine-tuning that actually moves weights, with the operational details that make long runs survivable.',
    points: [
      'LoRA and QLoRA, gradient accumulation, LR schedules, early stopping',
      'Checkpoint pruning, resume from interruption, NaN abort',
      'Modes for fine-tuning, agent traces, DPO preference alignment and scratch runs',
      'Nexara files compile straight into a hardware-aware training script',
    ],
  },
  {
    label: '03',
    title: 'Agents',
    body: 'Tool calls parsed out of model output and executed under a security layer, so a local model can do work without being handed the whole machine.',
    points: [
      'create_file, read_file, edit_file, run_command, web_request and more',
      'Workspace restrictions and consent prompts before anything touches disk',
      'Every action written to ~/.inferforge/audit.log, with backups',
      'An eval harness scores parseable calls, correct tools and argument accuracy',
    ],
  },
  {
    label: '04',
    title: 'Deployment',
    body: 'The newest piece. Take a model you trained on your own hardware and put it behind a script tag on a real website.',
    points: [
      'forge embedd --sdk publishes the model and hosts its SDK on the site',
      'One script tag mounts a working chat widget, no build step required',
      'Chat relays to the machine running forge serve, streaming end to end',
      'Per-model embed keys, so a published key only unlocks that one model',
    ],
  },
]

const tiers: Tier[] = [
  {
    name: 'Verified working',
    dot: 'bg-emerald-400',
    ring: 'border-emerald-400/25 bg-emerald-400/[0.06]',
    items: 'run / chat, registry, Ollama import, serve API, SDK embed, LoRA + DPO + agent training, merging, agent sandbox, Nexara compile to train',
  },
  {
    name: 'Beta-gated',
    dot: 'bg-amber-400',
    ring: 'border-amber-400/25 bg-amber-400/[0.06]',
    items: 'train, merge, nexara, benchmark, optimize, test, monitor, web, plugin, create, checkpoint, all behind the beta flag',
  },
  {
    name: 'Written, unproven',
    dot: 'bg-rose-400',
    ring: 'border-rose-400/25 bg-rose-400/[0.06]',
    items: 'WebGPU browser inference, distributed and federated training, the forever loop, cloud and remote, team registry, most Nexara auxiliary modules',
  },
]

const terminal: { prompt: string; out: string[] }[] = [
  { prompt: 'forge pull llama3:8b', out: ['resolved ollama tag -> 4.7 GB', 'linked weights, registered as llama3:8b'] },
  { prompt: 'forge train llama3:8b --finetune', out: ['LoRA r=16 · 3 epochs · early stop on', 'checkpoint saved -> llama3-8b-tuned'] },
  { prompt: 'forge embedd llama3-8b-tuned --sdk', out: ['published -> inferforge.org/sdk/llama3-8b-tuned.js'] },
]

export default function Home() {
  return (
    <div className="relative overflow-hidden">
      <div className="pointer-events-none absolute inset-x-0 top-0 h-[720px] bg-[radial-gradient(ellipse_60%_50%_at_50%_0%,rgba(251,191,36,0.13),transparent_70%)]" />
      <div className="pointer-events-none absolute inset-x-0 top-0 h-[720px] bg-grid-pattern [background-size:52px_52px] opacity-[0.35] [mask-image:linear-gradient(to_bottom,black,transparent)]" />

      <section className="relative max-w-6xl mx-auto px-6 pt-20 pb-24 md:pt-28 md:pb-32">
        <div className="max-w-3xl">
          <span className="inline-flex items-center gap-2 rounded-full border border-amber-400/25 bg-amber-400/10 px-3 py-1 text-[12px] font-semibold text-amber-200">
            <Zap className="h-3.5 w-3.5" />
            Self-hosted · your hardware · your weights
          </span>

          <h1 className="mt-6 text-[42px] leading-[1.05] font-extrabold tracking-tight md:text-[64px]">
            Run, train and ship
            <br />
            <span className="bg-gradient-to-r from-amber-200 via-amber-400 to-orange-400 bg-clip-text text-transparent">
              your own AI models.
            </span>
          </h1>

          <p className="mt-6 max-w-2xl text-[17px] leading-relaxed text-white/60 md:text-[19px]">
            InferForge is a local AI platform and CLI for running, training, merging and deploying
            large language models on your own machine. One <span className="font-mono text-amber-200">forge</span> command
            covers the whole path: a self-hosted model runner, a fine-tuning engine and an agent
            runtime in a single tool.
          </p>

          <div className="mt-9 flex flex-col gap-3 sm:flex-row sm:items-center">
            <Link
              to="/download"
              className="inline-flex items-center justify-center gap-2 rounded-xl bg-amber-400 px-6 py-3.5 text-[15px] font-semibold text-[#0e1b3d] transition hover:bg-amber-300"
            >
              Get InferForge
              <ArrowRight className="h-4 w-4" />
            </Link>
            <Link
              to="/docs"
              className="inline-flex items-center justify-center gap-2 rounded-xl border border-white/15 px-6 py-3.5 text-[15px] font-semibold text-white/85 transition hover:border-white/30 hover:bg-white/5"
            >
              Read the docs
            </Link>
          </div>

          <div className="mt-8 flex max-w-md items-center gap-2 rounded-xl border border-white/10 bg-black/25 px-4 py-3">
            <span className="select-none font-mono text-[13px] text-amber-300/70">$</span>
            <code className="flex-1 truncate font-mono text-[13px] text-white/80">pip install inferforge</code>
            <CopyableCommand command="pip install inferforge" />
          </div>
        </div>

        <div className="mt-16 overflow-hidden rounded-2xl border border-white/10 bg-black/35 shadow-2xl shadow-black/40">
          <div className="flex items-center gap-2 border-b border-white/10 px-4 py-3">
            <span className="h-2.5 w-2.5 rounded-full bg-rose-400/70" />
            <span className="h-2.5 w-2.5 rounded-full bg-amber-400/70" />
            <span className="h-2.5 w-2.5 rounded-full bg-emerald-400/70" />
            <span className="ml-3 flex items-center gap-1.5 text-[12px] font-medium text-white/35">
              <Terminal className="h-3.5 w-3.5" />
              inferforge
            </span>
          </div>
          <div className="space-y-4 px-5 py-6 font-mono text-[13px] leading-relaxed md:px-7">
            {terminal.map((line) => (
              <div key={line.prompt}>
                <div className="flex gap-2">
                  <span className="select-none text-amber-300/70">$</span>
                  <span className="text-white/90">{line.prompt}</span>
                </div>
                {line.out.map((o) => (
                  <div key={o} className="pl-5 text-white/40">
                    {o}
                  </div>
                ))}
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="relative border-t border-white/[0.07] bg-white/[0.015]">
        <div className="max-w-6xl mx-auto px-6 py-20 md:py-24">
          <div className="max-w-2xl">
            <h2 className="text-[30px] font-bold tracking-tight md:text-[38px]">What InferForge does</h2>
            <p className="mt-4 text-[16px] leading-relaxed text-white/55">
              Six things, one CLI. Every model stays on hardware you control, and nothing is sent
              anywhere you did not point it at yourself.
            </p>
          </div>

          <div className="mt-12 grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
            {capabilities.map((c) => (
              <div
                key={c.title}
                className="group rounded-2xl border border-white/10 bg-white/[0.035] p-6 transition hover:-translate-y-0.5 hover:border-amber-400/30 hover:bg-white/[0.06]"
              >
                <span className="inline-flex h-10 w-10 items-center justify-center rounded-xl border border-amber-400/20 bg-amber-400/10 text-amber-300">
                  <c.icon className="h-5 w-5" />
                </span>
                <h3 className="mt-5 text-[17px] font-semibold tracking-tight">{c.title}</h3>
                <code className="mt-1.5 block font-mono text-[12px] text-amber-300/70">{c.command}</code>
                <p className="mt-3 text-[14px] leading-relaxed text-white/50">{c.body}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="relative">
        <div className="max-w-6xl mx-auto px-6 py-20 md:py-24">
          <div className="max-w-2xl">
            <h2 className="text-[30px] font-bold tracking-tight md:text-[38px]">The stack</h2>
            <p className="mt-4 text-[16px] leading-relaxed text-white/55">
              Around forty-four thousand lines of Python organised around four jobs: run models,
              train them, give them tools, and put them on the web.
            </p>
          </div>

          <div className="mt-12 grid gap-5 lg:grid-cols-2">
            {pillars.map((p) => (
              <div key={p.title} className="rounded-2xl border border-white/10 bg-white/[0.035] p-7">
                <div className="flex items-baseline gap-3">
                  <span className="font-mono text-[13px] font-semibold text-amber-300/60">{p.label}</span>
                  <h3 className="text-[20px] font-semibold tracking-tight">{p.title}</h3>
                </div>
                <p className="mt-3 text-[14px] leading-relaxed text-white/50">{p.body}</p>
                <ul className="mt-5 space-y-2.5">
                  {p.points.map((point) => (
                    <li key={point} className="flex gap-3 text-[14px] leading-relaxed text-white/65">
                      <Check className="mt-[3px] h-4 w-4 shrink-0 text-amber-300/70" />
                      <span>{point}</span>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="relative border-t border-white/[0.07] bg-white/[0.015]">
        <div className="max-w-6xl mx-auto px-6 py-20 md:py-24">
          <div className="grid gap-12 lg:grid-cols-[1fr_1.15fr] lg:gap-16">
            <div>
              <span className="inline-flex items-center gap-2 rounded-full border border-white/12 bg-white/5 px-3 py-1 text-[12px] font-semibold text-white/60">
                <Shield className="h-3.5 w-3.5" />
                Honest status
              </span>
              <h2 className="mt-5 text-[30px] font-bold tracking-tight md:text-[38px]">
                What actually works
              </h2>
              <p className="mt-4 text-[16px] leading-relaxed text-white/55">
                This is a solo project with the surface area of a startup&apos;s entire stack, so
                there is a real gap between what is written and what is verified. Rather than hide
                it, here is the split.
              </p>
              <Link
                to="/docs"
                className="mt-7 inline-flex items-center gap-2 text-[14px] font-semibold text-amber-300 transition hover:text-amber-200"
              >
                See the full command reference
                <ArrowRight className="h-4 w-4" />
              </Link>
            </div>

            <div className="space-y-4">
              {tiers.map((t) => (
                <div key={t.name} className={`rounded-2xl border p-6 ${t.ring}`}>
                  <div className="flex items-center gap-2.5">
                    <span className={`h-2 w-2 rounded-full ${t.dot}`} />
                    <h3 className="text-[15px] font-semibold tracking-tight">{t.name}</h3>
                  </div>
                  <p className="mt-3 font-mono text-[13px] leading-relaxed text-white/50">{t.items}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      <section className="relative">
        <div className="max-w-6xl mx-auto px-6 py-20 md:py-24">
          <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
            {[
              { icon: Layers, value: '38', label: 'models tracked by the local registry' },
              { icon: Server, value: ':11435', label: 'OpenAI-compatible API port' },
              { icon: Cpu, value: '44k', label: 'lines of Python behind one CLI' },
              { icon: Terminal, value: '50+', label: 'forge commands across the toolchain' },
            ].map((s) => (
              <div key={s.label} className="rounded-2xl border border-white/10 bg-white/[0.035] p-6">
                <s.icon className="h-5 w-5 text-amber-300/70" />
                <div className="mt-4 font-mono text-[26px] font-bold tracking-tight">{s.value}</div>
                <div className="mt-1.5 text-[13px] leading-relaxed text-white/45">{s.label}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="relative border-t border-white/[0.07]">
        <div className="max-w-6xl mx-auto px-6 py-20 md:py-28">
          <div className="relative overflow-hidden rounded-3xl border border-amber-400/20 bg-gradient-to-br from-amber-400/[0.12] via-white/[0.03] to-transparent px-8 py-14 text-center md:px-16 md:py-20">
            <h2 className="text-[30px] font-bold tracking-tight md:text-[42px]">
              Your models. Your machine.
            </h2>
            <p className="mx-auto mt-5 max-w-xl text-[16px] leading-relaxed text-white/60">
              Install the CLI, pull a model, and have it answering on your own site by the end of the
              afternoon. Create an account to publish models and manage your embed keys.
            </p>
            <div className="mt-9 flex flex-col justify-center gap-3 sm:flex-row">
              <Link
                to="/create-account"
                className="inline-flex items-center justify-center gap-2 rounded-xl bg-amber-400 px-7 py-3.5 text-[15px] font-semibold text-[#0e1b3d] transition hover:bg-amber-300"
              >
                Create account
                <ArrowRight className="h-4 w-4" />
              </Link>
              <Link
                to="/download"
                className="inline-flex items-center justify-center rounded-xl border border-white/15 px-7 py-3.5 text-[15px] font-semibold text-white/85 transition hover:border-white/30 hover:bg-white/5"
              >
                Download the CLI
              </Link>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}
