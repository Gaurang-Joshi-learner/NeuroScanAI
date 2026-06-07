import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import { Brain, Eye, EyeOff, AlertCircle } from 'lucide-react'

export default function Register() {
  const { register } = useAuth()
  const navigate = useNavigate()
  const [form, setForm] = useState({ email:'', password:'', full_name:'', org_name:'' })
  const [showPw, setShowPw] = useState(false)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const handleSubmit = async e => {
    e.preventDefault(); setError(''); setLoading(true)
    try { await register(form.email, form.password, form.full_name, form.org_name); navigate('/dashboard') }
    catch (err) { setError(err.response?.data?.detail || 'Registration failed') }
    finally { setLoading(false) }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-950 px-4 py-12">
      <div className="w-full max-w-md">
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-14 h-14 bg-violet-600 rounded-2xl mb-4">
            <Brain className="w-8 h-8 text-white" />
          </div>
          <h1 className="text-3xl font-bold text-white">NeuroScanAI</h1>
          <p className="text-gray-400 mt-1">Create your research account</p>
        </div>
        <div className="card">
          <h2 className="text-xl font-semibold text-white mb-6">Create account</h2>
          {error && <div className="flex items-center gap-2 p-3 bg-red-900/30 border border-red-700 rounded-lg mb-4 text-red-400 text-sm"><AlertCircle className="w-4 h-4 flex-shrink-0" />{error}</div>}
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-1">Full Name</label>
                <input type="text" required className="input-modern" placeholder="Dr. Jane Smith"
                  value={form.full_name} onChange={e => setForm({...form, full_name: e.target.value})} />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-1">Lab / Org</label>
                <input type="text" required className="input-modern" placeholder="MIT BCI Lab"
                  value={form.org_name} onChange={e => setForm({...form, org_name: e.target.value})} />
              </div>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-1">Email</label>
              <input type="email" required className="input-modern" placeholder="you@lab.edu"
                value={form.email} onChange={e => setForm({...form, email: e.target.value})} />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-1">Password</label>
              <div className="relative">
                <input type={showPw ? 'text' : 'password'} required className="input-modern pr-10"
                  value={form.password} onChange={e => setForm({...form, password: e.target.value})} />
                <button type="button" onClick={() => setShowPw(!showPw)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500">
                  {showPw ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
              <p className="text-xs text-gray-500 mt-1">Min 8 chars, 1 uppercase, 1 number</p>
            </div>
            <button type="submit" disabled={loading} className="btn-primary w-full py-2.5">
              {loading ? 'Creating account...' : 'Create account'}
            </button>
          </form>
          <p className="text-center text-sm text-gray-500 mt-4">
            Already registered? <Link to="/login" className="text-violet-400 hover:underline">Sign in</Link>
          </p>
        </div>
      </div>
    </div>
  )
}
