import { useCallback, useEffect, useState } from 'react'
import { api, errorText } from '../../api/client'
import type { LlmProvider, LlmSettings, LlmSettingsPatch } from '../../api/types'
import { useNavGuard } from '../../shell/navGuard'
import styles from './SettingsPage.module.css'

const MAX_PROMPT = 4000
const DEFAULT_TEMPERATURE = 0.7
const LEAVE_MESSAGE = '有未儲存的變更，確定要離開？'
const PROVIDERS: { id: LlmProvider; label: string }[] = [
  { id: 'ollama', label: 'Ollama' },
  { id: 'gemini', label: 'Gemini' },
]

interface Draft {
  llm_provider: LlmProvider
  ollama_model: string
  gemini_model: string
  system_prompt: string
  temperature: number | null
}

type ModelsState =
  | { status: 'loading' }
  | { status: 'ready'; models: string[] }
  | { status: 'error'; message: string }
type ModelsResult = ModelsState & { key: string }

function toDraft(s: LlmSettings): Draft {
  return {
    llm_provider: s.llm_provider,
    ollama_model: s.ollama_model,
    gemini_model: s.gemini_model,
    system_prompt: s.system_prompt,
    temperature: s.temperature,
  }
}

function diff(saved: Draft, draft: Draft): LlmSettingsPatch {
  const patch: LlmSettingsPatch = {}
  if (draft.llm_provider !== saved.llm_provider) patch.llm_provider = draft.llm_provider
  if (draft.ollama_model !== saved.ollama_model) patch.ollama_model = draft.ollama_model
  if (draft.gemini_model !== saved.gemini_model) patch.gemini_model = draft.gemini_model
  if (draft.system_prompt !== saved.system_prompt) patch.system_prompt = draft.system_prompt
  if (draft.temperature !== saved.temperature) patch.temperature = draft.temperature
  return patch
}

export default function SettingsPage() {
  const { setBlocked } = useNavGuard()
  const [loaded, setLoaded] = useState<LlmSettings | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [loadAttempt, setLoadAttempt] = useState(0)
  const [saved, setSaved] = useState<Draft | null>(null)
  const [draft, setDraft] = useState<Draft | null>(null)
  const [modelsResult, setModelsResult] = useState<ModelsResult | null>(null)
  const [modelsAttempt, setModelsAttempt] = useState(0)
  const [saving, setSaving] = useState(false)
  const [saveError, setSaveError] = useState<string | null>(null)
  const [justSaved, setJustSaved] = useState(false)

  useEffect(() => {
    let cancelled = false
    api.getSettings().then(
      (s) => {
        if (cancelled) return
        setLoaded(s)
        setSaved(toDraft(s))
        setDraft(toDraft(s))
        setLoadError(null)
      },
      (e) => {
        if (!cancelled) setLoadError(errorText(e, '無法載入設定'))
      },
    )
    return () => {
      cancelled = true
    }
  }, [loadAttempt])

  const provider = draft?.llm_provider ?? null
  useEffect(() => {
    if (provider === null) return
    let cancelled = false
    const key = `${provider}:${modelsAttempt}`
    api.listModels(provider).then(
      (list) => {
        if (!cancelled) setModelsResult({ key, status: 'ready', models: list })
      },
      (e) => {
        if (!cancelled) setModelsResult({ key, status: 'error', message: errorText(e, '無法取得模型清單') })
      },
    )
    return () => {
      cancelled = true
    }
  }, [provider, modelsAttempt])

  // a result only counts for the provider/attempt it was fetched for
  const models: ModelsState =
    modelsResult !== null && modelsResult.key === `${provider}:${modelsAttempt}`
      ? modelsResult
      : { status: 'loading' }

  const patch = saved && draft ? diff(saved, draft) : {}
  const dirty = Object.keys(patch).length > 0

  useEffect(() => {
    setBlocked(dirty ? LEAVE_MESSAGE : null)
  }, [dirty, setBlocked])
  useEffect(() => () => setBlocked(null), [setBlocked])

  const update = useCallback((change: Partial<Draft>) => {
    setDraft((d) => (d ? { ...d, ...change } : d))
    setJustSaved(false)
    setSaveError(null)
  }, [])

  if (loadError !== null) {
    return (
      <div className={styles.page}>
        <p role="alert" className={styles.error}>
          {loadError}
        </p>
        <button
          type="button"
          className={styles.secondary}
          onClick={() => {
            setLoadError(null)
            setLoadAttempt((n) => n + 1)
          }}
        >
          重試
        </button>
      </div>
    )
  }
  if (!loaded || !draft || !saved) {
    return (
      <div className={styles.page} role="status">
        載入中…
      </div>
    )
  }

  const modelKey = draft.llm_provider === 'ollama' ? 'ollama_model' : 'gemini_model'
  const currentModel = draft[modelKey]
  const listed = models.status === 'ready' ? models.models : null
  const missing = listed !== null && !listed.includes(currentModel)
  const options = listed ?? []
  const promptTooLong = draft.system_prompt.length > MAX_PROMPT

  const save = async () => {
    if (!dirty || promptTooLong || saving) return
    setSaving(true)
    setSaveError(null)
    setJustSaved(false)
    try {
      const result = await api.saveSettings(patch)
      setLoaded(result)
      setSaved(toDraft(result))
      setDraft(toDraft(result))
      setJustSaved(true)
    } catch (e) {
      setSaveError(errorText(e, '儲存失敗'))
    } finally {
      setSaving(false)
    }
  }

  return (
    <form
      className={styles.page}
      onSubmit={(e) => {
        e.preventDefault()
        void save()
      }}
    >
      <h2 className={styles.title}>系統設定</h2>

      <fieldset className={styles.group}>
        <legend>LLM 供應商</legend>
        {PROVIDERS.map((p) => {
          const status = loaded.providers[p.id]
          return (
            <label key={p.id} className={styles.radio}>
              <input
                type="radio"
                name="llm_provider"
                checked={draft.llm_provider === p.id}
                disabled={!status.available}
                onChange={() => update({ llm_provider: p.id })}
              />
              <span>{p.label}</span>
              {!status.available && <span className={styles.hint}>{status.reason}</span>}
            </label>
          )
        })}
      </fieldset>

      <div className={styles.group}>
        <label htmlFor="settings-model">模型</label>
        <select
          id="settings-model"
          value={currentModel}
          disabled={models.status === 'loading'}
          onChange={(e) => update({ [modelKey]: e.target.value })}
        >
          {missing && <option value={currentModel}>{currentModel}（清單中找不到）</option>}
          {models.status !== 'ready' && <option value={currentModel}>{currentModel}</option>}
          {options.map((m) => (
            <option key={m} value={m}>
              {m}
            </option>
          ))}
        </select>
        {models.status === 'loading' && <span className={styles.hint}>載入中…</span>}
        {models.status === 'error' && (
          <div className={styles.inline}>
            <span className={styles.error}>{models.message}</span>
            <button
              type="button"
              className={styles.secondary}
              onClick={() => setModelsAttempt((n) => n + 1)}
            >
              重試
            </button>
          </div>
        )}
      </div>

      <div className={styles.group}>
        <label htmlFor="settings-prompt">系統提示詞</label>
        <textarea
          id="settings-prompt"
          rows={8}
          value={draft.system_prompt}
          onChange={(e) => update({ system_prompt: e.target.value })}
        />
        <span className={promptTooLong ? styles.error : styles.hint}>
          {draft.system_prompt.length} / {MAX_PROMPT}
        </span>
        {promptTooLong && <span className={styles.error}>系統提示詞不可超過 4000 字</span>}
      </div>

      <div className={styles.group}>
        <span id="settings-temp-label">temperature</span>
        <label className={styles.radio}>
          <input
            type="checkbox"
            checked={draft.temperature === null}
            onChange={(e) =>
              update({ temperature: e.target.checked ? null : DEFAULT_TEMPERATURE })
            }
          />
          <span>使用預設</span>
        </label>
        {draft.temperature !== null && (
          <div className={styles.inline}>
            <input
              type="range"
              aria-labelledby="settings-temp-label"
              min={0}
              max={2}
              step={0.1}
              value={draft.temperature}
              onChange={(e) => update({ temperature: Number(e.target.value) })}
            />
            <span>{draft.temperature.toFixed(1)}</span>
          </div>
        )}
      </div>

      <div className={styles.actions}>
        <button
          type="submit"
          className={styles.primary}
          disabled={!dirty || promptTooLong || saving}
        >
          儲存
        </button>
        {justSaved && (
          <span role="status" className={styles.ok}>
            已儲存
          </span>
        )}
        {saveError !== null && (
          <span role="alert" className={styles.error}>
            {saveError}
          </span>
        )}
      </div>
    </form>
  )
}
