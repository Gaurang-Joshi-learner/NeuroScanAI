import { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import api from '../lib/api'
import { ArrowLeft, Download, Loader, AlertCircle, Activity, Mic,
         RefreshCw, ChevronDown, ChevronUp, AlertTriangle } from 'lucide-react'
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, Cell } from 'recharts'
import { riskColor, riskBg } from '../lib/utils'


function SeizureResults({ result }) {
  const [showAll, setShowAll] = useState(false)
  const timeline = result.timeline || []
  const chartData = timeline.map(t => ({
    time: `${t.time_start}s`,
    prob: parseFloat((t.seizure_prob * 100).toFixed(1)),
    seizure: t.prediction === 'SEIZURE'
  }))
  const displayedTimeline = showAll ? timeline : timeline.slice(0, 20)

  return (
    <div className="space-y-6">
      {/* Risk summary */}
      <div className={`card border-2 ${riskBg(result.risk_level)}`}>
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm text-gray-400">Overall Risk Level</p>
            <p className={`text-4xl font-black mt-1 ${riskColor(result.risk_level)}`}>{result.risk_level}</p>
            <p className="text-gray-400 text-sm mt-1">
              Max probability: {(result.overall_risk_score * 100).toFixed(1)}% ·
              Mean: {(result.mean_risk_score * 100).toFixed(1)}% ·
              {result.n_seizure_epochs} seizure epoch{result.n_seizure_epochs !== 1 ? 's' : ''} detected
            </p>
          </div>
          <Activity className={`w-16 h-16 opacity-20 ${riskColor(result.risk_level)}`} />
        </div>
      </div>

      {/* Metrics grid */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        {[
          { label: 'Recording Duration', value: `${result.recording_duration_sec}s` },
          { label: 'Total Epochs',       value: result.n_epochs },
          { label: 'Seizure Epochs',     value: `${result.seizure_epoch_pct}%` },
          { label: 'EEG Channels',       value: result.n_channels },
        ].map(m => (
          <div key={m.label} className="card">
            <p className="text-xs text-gray-500 mb-1">{m.label}</p>
            <p className="text-3xl font-bold text-white">{m.value}</p>
          </div>
        ))}
      </div>

      {/* Seizure probability chart */}
      {chartData.length > 0 && (
        <div className="card hover:border-violet-500/20 transition-all">
          <h3 className="font-semibold text-white mb-4">Seizure Probability Timeline</h3>
          <ResponsiveContainer width="100%" height={220}>
            <AreaChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="time" tick={{ fill: '#9ca3af', fontSize: 11 }} interval={Math.floor(chartData.length/8)} />
              <YAxis tick={{ fill: '#9ca3af', fontSize: 11 }} domain={[0, 100]} tickFormatter={v => `${v}%`} />
              <Tooltip formatter={v => [`${v}%`, 'Seizure Prob']}
                       contentStyle={{ background: '#1f2937', border: '1px solid #374151', borderRadius: 8 }}
                       labelStyle={{ color: '#e5e7eb' }} />
              <Area type="monotone" dataKey="prob" stroke="#ef4444" fill="#ef444430"
                    strokeWidth={2} dot={false} />
            </AreaChart>
          </ResponsiveContainer>
          <div className="flex items-center gap-4 mt-2 text-xs text-gray-500">
            <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-red-400 inline-block" /> Above 50% = SEIZURE prediction</span>
          </div>
        </div>
      )}

      {/* Epoch table */}
      <div className="card p-0 overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-800 flex items-center justify-between">
          <h3 className="font-semibold text-white">Epoch-by-Epoch Results ({timeline.length} total)</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-gray-800/50">
              <tr>
                {['Epoch','Start','End','Probability','Prediction'].map(h => (
                  <th key={h} className="px-4 py-3 text-left text-xs font-semibold text-gray-400 uppercase">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800">
              {displayedTimeline.map(t => (
                <tr key={t.epoch_idx} className={`hover:bg-gray-800/30 ${t.prediction === 'SEIZURE' ? 'bg-red-900/10' : ''}`}>
                  <td className="px-4 py-2 text-gray-300">{t.epoch_idx}</td>
                  <td className="px-4 py-2 text-gray-300">{t.time_start}s</td>
                  <td className="px-4 py-2 text-gray-300">{t.time_end}s</td>
                  <td className="px-4 py-2">
                    <div className="flex items-center gap-2">
                      <div className="w-20 h-1.5 bg-gray-700 rounded-full">
                        <div className="h-1.5 rounded-full bg-red-400" style={{ width: `${t.seizure_prob*100}%` }} />
                      </div>
                      <span className="text-gray-300 text-xs">{(t.seizure_prob*100).toFixed(1)}%</span>
                    </div>
                  </td>
                  <td className="px-4 py-2">
                    <span className={`text-xs font-bold ${t.prediction === 'SEIZURE' ? 'text-red-400' : 'text-green-400'}`}>
                      {t.prediction}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {timeline.length > 20 && (
          <button onClick={() => setShowAll(!showAll)}
            className="w-full py-3 text-sm text-violet-400 hover:bg-gray-800/30 transition-colors flex items-center justify-center gap-1">
            {showAll ? <><ChevronUp className="w-4 h-4" />Show less</>
                     : <><ChevronDown className="w-4 h-4" />Show all {timeline.length} epochs</>}
          </button>
        )}
      </div>
    </div>
  )
}

function SpeechResults({ result }) {
  const trials = result.trials || []
  const correct = trials.filter(t => t.correct === true).length
  const accuracy = result.accuracy || 0
  const chance = result.chance_level || 0

  const wordCounts = {}
  trials.forEach(t => { wordCounts[t.predicted_word] = (wordCounts[t.predicted_word] || 0) + 1 })
  const chartData = Object.entries(wordCounts).map(([word, count]) => ({ word, count }))

  return (
    <div className="space-y-6">
      {/* Accuracy summary */}
      <div className="card border-2 border-violet-700 bg-violet-900/20">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm text-gray-400">Decoding Accuracy</p>
            <p className="text-4xl font-black mt-1 text-violet-300">{(accuracy*100).toFixed(1)}%</p>
            <p className="text-gray-400 text-sm mt-1">
              Chance level: {(chance*100).toFixed(1)}% ·
              {correct} / {trials.length} trials correct ·
              {result.n_classes} classes
            </p>
          </div>
          <Mic className="w-16 h-16 text-violet-400 opacity-20" />
        </div>
        <div className="mt-3 w-full bg-gray-800 rounded-full h-2">
          <div className="bg-violet-500 h-2 rounded-full" style={{ width: `${accuracy*100}%` }} />
        </div>
        <div className="flex justify-between text-xs text-gray-500 mt-1">
          <span>0%</span>
          <span className="text-gray-400">Chance: {(chance*100).toFixed(1)}%</span>
          <span>100%</span>
        </div>
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        {[
          { label: 'Total Trials',   value: result.n_trials },
          { label: 'Correct',        value: correct },
          { label: 'Classes',        value: result.n_classes },
          { label: 'Above Chance',   value: accuracy > chance ? `+${((accuracy-chance)*100).toFixed(1)}%` : 'No' },
        ].map(m => (
          <div key={m.label} className="card">
            <p className="text-xs text-gray-500 mb-1">{m.label}</p>
            <p className="text-xl font-bold text-white">{m.value}</p>
          </div>
        ))}
      </div>

      {/* Prediction distribution */}
      {chartData.length > 0 && (
        <div className="card">
          <h3 className="font-semibold text-white mb-4">Prediction Distribution</h3>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="word" tick={{ fill: '#9ca3af', fontSize: 11 }} />
              <YAxis tick={{ fill: '#9ca3af', fontSize: 11 }} />
              <Tooltip contentStyle={{ background: '#1f2937', border: '1px solid #374151', borderRadius: 8 }}
                       labelStyle={{ color: '#e5e7eb' }} />
              <Bar dataKey="count" radius={[4,4,0,0]}>
                {chartData.map((_, i) => <Cell key={i} fill="#7c3aed" />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Trial table */}
      <div className="card p-0 overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-800">
          <h3 className="font-semibold text-white">Trial Results ({trials.length} trials)</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-gray-800/50">
              <tr>
                {['Trial','Predicted','True','Correct','Top-3 Predictions'].map(h => (
                  <th key={h} className="px-4 py-3 text-left text-xs font-semibold text-gray-400 uppercase">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800">
              {trials.slice(0,50).map(t => (
                <tr key={t.trial_idx} className={`hover:bg-gray-800/30 ${t.correct === true ? 'bg-green-900/5' : t.correct === false ? 'bg-red-900/5' : ''}`}>
                  <td className="px-4 py-2 text-gray-400">{t.trial_idx}</td>
                  <td className="px-4 py-2 font-mono text-violet-300">{t.predicted_word}</td>
                  <td className="px-4 py-2 font-mono text-gray-300">{t.true_word || '—'}</td>
                  <td className="px-4 py-2">
                    {t.correct === null ? '—'
                      : <span className={`text-xs font-bold ${t.correct ? 'text-green-400' : 'text-red-400'}`}>
                          {t.correct ? '✓' : '✗'}
                        </span>
                    }
                  </td>
                  <td className="px-4 py-2 text-xs text-gray-500">
                    {t.top3?.map(p => `${p.word} ${(p.confidence*100).toFixed(0)}%`).join(' · ')}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

export default function Results() {
  const { id } = useParams()
  const navigate = useNavigate()
  const [analysis, setAnalysis] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [polling, setPolling] = useState(false)

  const fetchAnalysis = async () => {
    try {
      const { data } = await api.get(`/analysis/${id}`)
      setAnalysis(data)
    } catch (e) { setError('Failed to load analysis') }
    finally { setLoading(false) }
  }

  useEffect(() => { fetchAnalysis() }, [id])

  useEffect(() => {
    if (!analysis) return
    if (analysis.status === 'processing' || analysis.status === 'queued') {
      setPolling(true)
      const t = setInterval(fetchAnalysis, 3000)
      return () => clearInterval(t)
    } else { setPolling(false) }
  }, [analysis?.status])

  const downloadJson = () => {
    const blob = new Blob([JSON.stringify(analysis?.result, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a'); a.href = url
    a.download = `${analysis?.filename}_results.json`; a.click()
  }
const downloadPDF = async () => {
  try {
    const response = await api.get(
      `/analysis/${id}/pdf`,
      {
        responseType: "blob"
      }
    )

    const url = window.URL.createObjectURL(
      new Blob([response.data])
    )

    const link = document.createElement("a")

    link.href = url

    link.setAttribute(
      "download",
      `${analysis.filename}_report.pdf`
    )

    document.body.appendChild(link)

    link.click()

    link.remove()
  } catch (err) {
    console.error(err)
    alert("Failed to download PDF")
  }
}
  if (loading) return (
    <div className="flex items-center justify-center min-h-[60vh]">
      <Loader className="w-8 h-8 text-violet-500 animate-spin" />
    </div>
  )

  if (error) return (
    <div className="max-w-xl mx-auto px-4 py-16 text-center">
      <AlertCircle className="w-12 h-12 text-red-400 mx-auto mb-3" />
      <p className="text-gray-300">{error}</p>
      <button onClick={() => navigate('/dashboard')} className="btn-secondary text-white mt-4">Back</button>
    </div>
  )

  if (analysis?.status !== 'done') return (
    <div className="max-w-xl mx-auto px-4 py-16 text-center">
      <div className="w-16 h-16 bg-violet-900/30 rounded-2xl flex items-center justify-center mx-auto mb-4">
        {analysis?.status === 'failed'
          ? <AlertCircle className="w-8 h-8 text-red-400" />
          : <Loader className="w-8 h-8 text-violet-400 animate-spin" />}
      </div>
      <h2 className="text-xl font-bold text-white mb-2">
        {analysis?.status === 'failed' ? 'Analysis Failed' : 'Processing EEG...'}
      </h2>
      <p className="text-gray-400">
        {analysis?.status === 'failed'
          ? analysis?.error_message || 'An error occurred'
          : 'Running pipeline. This takes 30–120 seconds depending on file size.'}
      </p>
      {polling && (
        <p className="text-xs text-violet-400 mt-3 flex items-center justify-center gap-1">
          <RefreshCw className="w-3 h-3 animate-spin" />Auto-refreshing...
        </p>
      )}
      <button onClick={() => navigate('/dashboard')} className="btn-secondary mt-6">Back to Dashboard</button>
    </div>
  )

  const isSeizure = analysis.analysis_type === 'seizure'

  return (
    <div className="max-w-5xl mx-auto px-4 py-8">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <button onClick={() => navigate('/dashboard')}
            className="p-2 hover:bg-gray-800 rounded-lg text-gray-400 transition-colors">
            <ArrowLeft className="w-5 h-5" />
          </button>
          <div>
            <div className="flex items-center gap-2">
              {isSeizure
                ? <Activity className="w-5 h-5 text-red-400" />
                : <Mic className="w-5 h-5 text-violet-400" />}
              <h1 className="text-xl font-bold text-white">{analysis.filename}</h1>
            </div>
            <p className="text-sm text-gray-400 mt-0.5">
              {isSeizure ? 'Seizure Detection' : 'Speech Decoding'} ·
              Model: {analysis.model_version}
            </p>
          </div>
        </div>
<div className="flex gap-2">
  <button
    onClick={downloadPDF}
    className="btn-primary text-white flex items-center gap-2"
  >
    <Download className="w-4 h-4" />
    Download PDF
  </button>

  <button
    onClick={downloadJson}
    className="btn-secondary text-white flex items-center gap-2"
  >
    <Download className="w-4 h-4" />
    Download JSON
  </button>
</div>
      </div>

      {/* Clinical disclaimer */}
      <div className="flex items-start gap-2 p-3 bg-yellow-900/20 border border-yellow-700 rounded-lg mb-6">
        <AlertTriangle className="w-4 h-4 text-yellow-400 flex-shrink-0 mt-0.5" />
        <p className="text-xs text-yellow-300">
          For research use only. Results are not validated for clinical use and must not be used for medical decisions.
        </p>
      </div>

      {isSeizure
        ? <SeizureResults result={analysis.result} />
        : <SpeechResults result={analysis.result} />}
    </div>
  )
}
