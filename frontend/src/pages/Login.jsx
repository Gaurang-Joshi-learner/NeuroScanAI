import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import { Brain, Eye, EyeOff, AlertCircle } from 'lucide-react'

export default function Login() {
  const { login } = useAuth()
  const navigate = useNavigate()
  const [form, setForm] = useState({ email: '', password: '' })
  const [showPw, setShowPw] = useState(false)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const handleSubmit = async e => {
    e.preventDefault(); setError(''); setLoading(true)
    try { await login(form.email, form.password); navigate('/dashboard') }
    catch (err) { setError(err.response?.data?.detail || 'Login failed') }
    finally { setLoading(false) }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-950 px-4">
      <div className="w-full max-w-md">
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-14 h-14 bg-violet-600 rounded-2xl mb-4">
            <Brain className="w-8 h-8 text-white" />
          </div>
          <h1 className="text-3xl font-bold text-white">NeuroScanAI</h1>
          <p className="text-gray-400 mt-1">EEG Analysis Platform</p>
        </div>
        <div className="card">
          <h2 className="text-xl font-semibold text-white mb-6">Sign in</h2>
          <button onClick={() => window.location.href = `${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/auth/google`}
            className="w-full flex items-center justify-center gap-3 px-4 py-2.5 border border-gray-700 rounded-lg hover:bg-gray-800 transition-colors mb-4 text-gray-200">
            <svg className="w-5 h-5" viewBox="0 0 24 24">
              <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
              <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
              <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
              <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
            </svg>
            Continue with Google
          </button>
          <div className="flex items-center gap-3 my-4"><div className="flex-1 h-px bg-gray-700" /><span className="text-xs text-gray-500">or</span><div className="flex-1 h-px bg-gray-700" /></div>
          {error && <div className="flex items-center gap-2 p-3 bg-red-900/30 border border-red-700 rounded-lg mb-4 text-red-400 text-sm"><AlertCircle className="w-4 h-4 flex-shrink-0" />{error}</div>}
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-1">Email</label>
              <input
  type="email"
  required
  className="input-modern"
  placeholder="you@lab.edu"
                value={form.email} onChange={e => setForm({...form, email: e.target.value})} />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-1">Password</label>
              <div className="relative">
                <input
  type={showPw ? 'text' : 'password'}
  required
  className="input-modern pr-10"
                  value={form.password} onChange={e => setForm({...form, password: e.target.value})} />
                <button type="button" onClick={() => setShowPw(!showPw)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500">
                  {showPw ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
            </div>
            <button type="submit" disabled={loading} className="btn-primary text-white w-full py-2.5">
              {loading ? 'Signing in...' : 'Sign in'}
            </button>
          </form>
          <p className="text-center text-sm text-gray-500 mt-4">
            No account? <Link to="/register" className="text-violet-400 hover:underline">Register</Link>
          </p>
        </div>
      </div>
    </div>
  )
}
