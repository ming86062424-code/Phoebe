import { useState } from 'react'
import ChartsTab from './ChartsTab'
import EventsTab from './EventsTab'
import ResourcesTab from './ResourcesTab'
import styles from './Monitor.module.css'

const TABS = [
  { id: 'events', label: '事件紀錄' },
  { id: 'resources', label: '資源使用' },
  { id: 'charts', label: '圖表' },
] as const
type TabId = (typeof TABS)[number]['id']

export default function MonitorPage() {
  const [tab, setTab] = useState<TabId>('events')
  return (
    <div className={styles.page}>
      <h1 className={styles.title}>監控</h1>
      <div className={styles.tabs} role="tablist" aria-label="監控">
        {TABS.map((t) => (
          <button
            key={t.id}
            type="button"
            role="tab"
            id={`monitor-tab-${t.id}`}
            aria-selected={tab === t.id}
            aria-controls={`monitor-panel-${t.id}`}
            className={tab === t.id ? styles.tabActive : styles.tab}
            onClick={() => setTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </div>
      <div role="tabpanel" id={`monitor-panel-${tab}`} aria-labelledby={`monitor-tab-${tab}`}>
        {tab === 'events' && <EventsTab />}
        {tab === 'resources' && <ResourcesTab />}
        {tab === 'charts' && <ChartsTab />}
      </div>
    </div>
  )
}
