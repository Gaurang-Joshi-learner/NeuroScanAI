import { useState, useEffect } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import api from '../lib/api'
import { useAuth } from '../context/AuthContext'
import { Brain, Upload, CheckCircle, XCircle, Loader, Clock, Trash2, Eye, Activity, Mic } from 'lucide-react'
import { formatDistanceToNow, riskColor, riskBg } from '../lib/utils'

const STATUS = {
  done:       { icon: CheckCircle, color: 'text-green-400',  bg: 'bg-green-900/20 border-green-800',  label: 'Done' },
  processing: { icon: Loader,       color: 'text-violet-400', bg: 'bg-violet-900/20 border-violet-800', label: 'Processing' },
  queued:     { icon: Clock,        color: 'text-yellow-400', bg: 'bg-yellow-900/20 border-yellow-800', label: 'Queued' },
  failed:     { icon: XCircle,      color: 'text-red-400',    bg: 'bg-red-900/20 border-red-800',       label: 'Failed' },
}

export default function Dashboard() {
  const { user } = useAuth()
  const navigate = useNavigate()
  const [analyses, setAnalyses] = useState([])
  const [loading, setLoading] = useState(true)
  const [deleting, setDeleting] = useState(null)

  const fetch = async () => {
    try { const { data } = await api.get('/analysis/'); setAnalyses(data) }
    catch (e) { console.error(e) }
    finally { setLoading(false) }
  }

  useEffect(() => {
    fetch()
    const iv = setInterval(() => {
      if (analyses.some(a => a.status === 'processing' || a.status === 'queued')) fetch()
    }, 4000)
    return () => clearInterval(iv)
  }, [analyses.length])

  const handleDelete = async (id, e) => {
    e.stopPropagation()
    if (!confirm('Delete this analysis?')) return
    setDeleting(id)
    try { await api.delete(`/analysis/${id}`); setAnalyses(prev => prev.filter(a => a.id !== id)) }
    finally { setDeleting(null) }
  }

  const seizureCount = analyses.filter(a => a.analysis_type === 'seizure').length
  const speechCount  = analyses.filter(a => a.analysis_type === 'speech').length
  const doneCount    = analyses.filter(a => a.status === 'done').length

  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold text-white">Welcome, {user?.full_name?.split(' ')[0] || 'Researcher'}</h1>
          <p className="text-gray-400 mt-1">EEG analysis history for {user?.email}</p>
        </div>
        <Link to="/analyze" className="btn-primary flex items-center gap-2">
          <Upload className="w-4 h-4" />New Analysis
        </Link>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-8">
        {[
          { label: 'Seizure Analyses', value: seizureCount, icon: Activity, color: 'red' },
          { label: 'Speech Analyses',  value: speechCount,  icon: Mic,      color: 'violet' },
          { label: 'Completed',        value: doneCount,    icon: CheckCircle, color: 'green' },
        ].map(s => (
          <div key={s.label}
     className="card hover:border-violet-500/30 transition-all duration-300 flex items-center gap-4">
            <div className={`w-12 h-12 rounded-xl bg-${s.color}-900/30 flex items-center justify-center`}>
              <s.icon className={`w-6 h-6 text-${s.color}-400`} />
            </div>
            <div>
              <p className="text-4xl font-bold text-white">{s.value}</p>
              <p className="text-sm text-gray-400">{s.label}</p>
            </div>
          </div>
        ))}
      </div>

      {/* Analyses list */}
      <div className="card p-0 overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-800">
          <h2 className="font-semibold text-white">All Analyses</h2>
        </div>
        {loading ? (
          <div className="flex items-center justify-center py-16">
            <Loader className="w-8 h-8 text-violet-500 animate-spin" />
          </div>
        ) : analyses.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-16 text-center">
            <Brain className="w-12 h-12 text-gray-700 mb-3" />
            <p className="text-gray-400 font-medium">No analyses yet</p>
            <p className="text-gray-600 text-sm mb-4">Upload an EDF or MAT file to get started</p>
            <Link to="/analyze" className="btn-primary flex items-center gap-2">
              <Upload className="w-4 h-4" />Upload EEG File
            </Link>
          </div>
        ) : (
          <div className="divide-y divide-gray-800">
            {analyses.map(a => {
              const cfg = STATUS[a.status] || STATUS.queued
              const Icon = cfg.icon
              const isSeizure = a.analysis_type === 'seizure'
              const riskLevel = a.result?.risk_level
              return (
                <div key={a.id}
                  onClick={() => a.status === 'done' && navigate(`/results/${a.id}`)}
                  className={`flex items-center gap-4 px-6 py-4 hover:bg-gray-800/50 transition-colors ${a.status === 'done' ? 'cursor-pointer' : ''}`}>
                  <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${isSeizure ? 'bg-red-900/30' : 'bg-violet-900/30'}`}>
                    {isSeizure ? <Activity className="w-5 h-5 text-red-400" /> : <Mic className="w-5 h-5 text-violet-400" />}
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="font-medium text-white truncate">{a.filename}</p>
                    <p className="text-sm text-gray-500">
                      {isSeizure ? 'Seizure Detection' : 'Speech Decoding'} · {formatDistanceToNow(a.created_at)}
                    </p>
                  </div>
                  {riskLevel && (
                    <span className={`text-xs font-bold px-2 py-1 rounded border ${riskBg(riskLevel)} ${riskColor(riskLevel)}`}>
                      {riskLevel}
                    </span>
                  )}
                  <div className={`flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium border ${cfg.bg} ${cfg.color}`}>
                    <Icon className={`w-3.5 h-3.5 ${a.status === 'processing' ? 'animate-spin' : ''}`} />
                    {cfg.label}
                  </div>
                  <div className="flex items-center gap-1">
                    {a.status === 'done' && (
                      <button onClick={e => { e.stopPropagation(); navigate(`/results/${a.id}`) }}
                        className="p-2 hover:bg-violet-900/30 rounded-lg text-violet-400 transition-colors">
                        <Eye className="w-4 h-4" />
                      </button>
                    )}
                    <button onClick={e => handleDelete(a.id, e)} disabled={deleting === a.id}
                      className="p-2 hover:bg-red-900/30 rounded-lg text-red-400 transition-colors">
                      {deleting === a.id ? <Loader className="w-4 h-4 animate-spin" /> : <Trash2 className="w-4 h-4" />}
                    </button>
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
}
