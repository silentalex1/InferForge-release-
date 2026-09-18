type Props = {
  id: string
  label: string
  type: string
  value: string
  placeholder: string
  autoComplete: string
  onChange: (value: string) => void
}

export default function AuthField({ id, label, type, value, placeholder, autoComplete, onChange }: Props) {
  return (
    <div>
      <label htmlFor={id} className="block text-[13px] font-medium text-white/60">
        {label}
      </label>
      <input
        id={id}
        type={type}
        value={value}
        placeholder={placeholder}
        autoComplete={autoComplete}
        onChange={(e) => onChange(e.target.value)}
        className="mt-2 w-full rounded-xl border border-white/12 bg-black/25 px-4 py-3 text-[15px] text-white placeholder:text-white/25 outline-none transition focus:border-amber-400/50 focus:ring-2 focus:ring-amber-400/15"
      />
    </div>
  )
}
