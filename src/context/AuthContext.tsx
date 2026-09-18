import { createContext, useContext, useEffect, useState, ReactNode } from 'react'

type User = { username: string; email: string }

type StoredUser = User & { password: string }

type AuthValue = {
  user: User | null
  register: (username: string, email: string, password: string) => { ok: boolean; error?: string }
  login: (username: string, password: string) => Promise<{ ok: boolean; error?: string }>
  resetPassword: (username: string, password: string, code: string) => Promise<{ ok: boolean; error?: string }>
  logout: () => void
  requestCode: (email: string) => string
}

const AuthContext = createContext<AuthValue>({
  user: null,
  register: () => ({ ok: false }),
  login: async () => ({ ok: false }),
  resetPassword: async () => ({ ok: false }),
  logout: () => {},
  requestCode: () => '',
})

function loadUsers(): StoredUser[] {
  try {
    const raw = localStorage.getItem('inferforge-users')
    return raw ? JSON.parse(raw) : []
  } catch {
    return []
  }
}

function saveUsers(users: StoredUser[]) {
  localStorage.setItem('inferforge-users', JSON.stringify(users))
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(() => {
    try {
      const raw = localStorage.getItem('inferforge-user')
      return raw ? JSON.parse(raw) : null
    } catch {
      return null
    }
  })

  useEffect(() => {
    if (user) localStorage.setItem('inferforge-user', JSON.stringify(user))
    else localStorage.removeItem('inferforge-user')
  }, [user])

  const requestCode = (email: string) => {
    const code = String(Math.floor(100000 + Math.random() * 900000))
    const expires = Date.now() + 5 * 60 * 1000
    localStorage.setItem('inferforge-pending-code', JSON.stringify({ email: email.toLowerCase(), code, expires }))
    return code
  }

  const register = (username: string, email: string, password: string) => {
    const users = loadUsers()
    const lowerU = username.toLowerCase()
    const lowerE = email.toLowerCase()
    if (users.some(u => u.username.toLowerCase() === lowerU)) return { ok: false, error: 'Username already taken.' }
    if (users.some(u => u.email.toLowerCase() === lowerE)) return { ok: false, error: 'Email already registered.' }
    const next: StoredUser = { username, email: lowerE, password }
    users.push(next)
    saveUsers(users)
    setUser({ username, email: lowerE })
    fetch("https://inferforge-email.asdwwas233.workers.dev/api/auth/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, email: lowerE, password }),
    }).catch(() => {})
    return { ok: true }
  }

  const login = async (username: string, password: string) => {
    const users = loadUsers()
    const lowerU = username.toLowerCase()
    const found = users.find(u => u.username.toLowerCase() === lowerU && u.password === password)
    if (found) {
      setUser({ username: found.username, email: found.email })
      return { ok: true }
    }
    try {
      const r = await fetch("https://inferforge-email.asdwwas233.workers.dev/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      })
      const j: any = await r.json().catch(() => ({}))
      if (r.ok && j?.ok && j.user) {
        const u = { username: j.user.username, email: j.user.email }
        const all = loadUsers()
        if (!all.some(x => x.username.toLowerCase() === u.username.toLowerCase())) {
          all.push({ ...u, password } as StoredUser)
          saveUsers(all)
        }
        setUser(u)
        return { ok: true }
      }
      return { ok: false, error: j?.error || 'Invalid username or password.' }
    } catch {
      return { ok: false, error: 'Network error. Check your connection.' }
    }
  }

  const resetPassword = async (username: string, password: string, code: string) => {
    const lowerU = username.toLowerCase()
    try {
      const r = await fetch("https://inferforge-email.asdwwas233.workers.dev/api/auth/reset", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username: lowerU, password, code }),
      })
      const j: any = await r.json().catch(() => ({}))
      if (!r.ok || !j?.ok) return { ok: false, error: j?.error || 'Could not reset the password.' }
    } catch {
      return { ok: false, error: 'Network error. Check your connection.' }
    }
    const users = loadUsers()
    const idx = users.findIndex(u => u.username.toLowerCase() === lowerU)
    if (idx >= 0) {
      users[idx] = { ...users[idx], password }
      saveUsers(users)
    }
    if (user && user.username.toLowerCase() === lowerU) setUser(null)
    return { ok: true }
  }

  const logout = () => setUser(null)

  return <AuthContext.Provider value={{ user, register, login, logout, requestCode, resetPassword }}>{children}</AuthContext.Provider>
}

export function useAuth() {
  return useContext(AuthContext)
}
