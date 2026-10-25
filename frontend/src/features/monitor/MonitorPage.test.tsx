import { act, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { api } from '../../api/client'
import type { MonitorEvent, MonitorSeries, MonitorStatus, ResourceSample } from '../../api/types'
import MonitorPage from './MonitorPage'

vi.mock('../../api/client', async (orig) => ({
  ...(await orig<typeof import('../../api/client')>()),
  api: {
    monitorEvents: vi.fn(),
    monitorStatus: vi.fn(),
    monitorSeries: vi.fn(),
    monitorSamples: vi.fn(),
  },
}))

const NOW = 1_800_000_000

function event(over: Partial<MonitorEvent> = {}): MonitorEvent {
  return {
    id: 1,
    kind: 'document',
    status: 'ok',
    started_at: NOW - 100,
    ended_at: NOW - 30,
    duration: 70,
    title: '草莓園入園方案.pdf',
    phases: { store: 0.4, parse: 63.2, embed: 6.4 },
    details: { parser: 'liteparse', pages: 12, chunks: 12, kb_id: 1, document_id: 9 },
    ...over,
  }
}

function sample(over: Partial<ResourceSample> = {}): ResourceSample {
  return {
    ts: NOW,
    source: 'phoebe-embedder-1',
    kind: 'container',
    project: 'phoebe',
    mem_mib: 2400,
    mem_limit_mib: 15923,
    cpu_pct: 12.5,
    gpu_util_pct: null,
    ...over,
  }
}

const status: MonitorStatus = {
  collector: { last_sample_at: NOW, interval_seconds: 5, error: '' },
  running: true,
  latest: [
    sample(),
    sample({ source: 'docker-es01-1', project: 'docker', mem_mib: 3700, cpu_pct: 0.2 }),
    sample({ source: 'GPU', kind: 'gpu', project: '', mem_mib: 3400, mem_limit_mib: 6144, cpu_pct: null, gpu_util_pct: 40 }),
  ],
}

const series: MonitorSeries = {
  since: NOW - 3600,
  until: NOW,
  step: 10,
  sources: [
    { source: 'phoebe-embedder-1', kind: 'container', project: 'phoebe', points: [
      { ts: NOW - 60, mem_mib: 2000, mem_limit_mib: 15923, cpu_pct: 5, gpu_util_pct: null },
      { ts: NOW - 50, mem_mib: 3000, mem_limit_mib: 15923, cpu_pct: 90, gpu_util_pct: null },
    ] },
    { source: 'docker-es01-1', kind: 'container', project: 'docker', points: [
      { ts: NOW - 60, mem_mib: 3700, mem_limit_mib: 15923, cpu_pct: 1, gpu_util_pct: null },
    ] },
    { source: 'GPU', kind: 'gpu', project: '', points: [
      { ts: NOW - 60, mem_mib: 3400, mem_limit_mib: 6144, cpu_pct: null, gpu_util_pct: 40 },
    ] },
  ],
  events: [event()],
}

beforeEach(() => {
  vi.mocked(api.monitorEvents).mockResolvedValue([
    event(),
    event({ id: 2, kind: 'chat', status: 'error', title: '今天天氣', phases: { first_token: 0.8 },
      details: { provider: 'gemini', model: 'gemini-2.5-flash', user: 'admin', error: '無法連線' } }),
  ])
  vi.mocked(api.monitorStatus).mockResolvedValue(status)
  vi.mocked(api.monitorSamples).mockResolvedValue([sample()])
  vi.mocked(api.monitorSeries).mockResolvedValue(series)
})
afterEach(() => {
  vi.useRealTimers()
  vi.clearAllMocks()
})

describe('MonitorPage', () => {
  it('lists task events with their phases, status and details', async () => {
    render(<MonitorPage />)
    expect(screen.getByRole('tab', { name: '事件紀錄' })).toHaveAttribute('aria-selected', 'true')
    const row = (await screen.findByText('草莓園入園方案.pdf')).closest('tr')!
    // phases in the order they happen
    expect(within(row).getAllByRole('listitem').map((li) => li.textContent)).toEqual([
      '解析 1 分 3 秒',
      '向量 6.4 秒',
      '寫入 400 ms',
    ])
    expect(within(row).getByText('1 分 10 秒')).toBeInTheDocument()
    expect(within(row).getByText('LiteParse・12 頁・12 個 chunk')).toBeInTheDocument()
    const chat = screen.getByText('今天天氣').closest('tr')!
    expect(within(chat).getByText('失敗')).toBeInTheDocument()
    expect(within(chat).getByText('無法連線')).toBeInTheDocument()
    expect(within(chat).getByText(/gemini gemini-2.5-flash/)).toBeInTheDocument()
  })

  it('filters events by kind', async () => {
    render(<MonitorPage />)
    await screen.findByText('草莓園入園方案.pdf')
    await userEvent.selectOptions(screen.getByRole('combobox', { name: '類型' }), 'chat')
    await waitFor(() => expect(api.monitorEvents).toHaveBeenLastCalledWith('chat'))
  })

  it('refreshes the events every 10 seconds', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true })
    render(<MonitorPage />)
    await screen.findByText('草莓園入園方案.pdf')
    const calls = vi.mocked(api.monitorEvents).mock.calls.length
    await act(() => vi.advanceTimersByTimeAsync(10000))
    expect(vi.mocked(api.monitorEvents).mock.calls.length).toBe(calls + 1)
  })

  it('shows the current usage, the total against the limit, and the samples', async () => {
    render(<MonitorPage />)
    await userEvent.click(screen.getByRole('tab', { name: '資源使用' }))
    expect(await screen.findByText(/monitor 服務運作中，每 5 秒取樣一次/)).toBeInTheDocument()
    const current = screen.getAllByRole('table')[0]
    expect(within(current).getByText('embedder')).toBeInTheDocument()
    expect(within(current).getByText('docker-es01-1')).toBeInTheDocument()
    const total = within(current).getByText('全部容器').closest('tr')!
    expect(total).toHaveTextContent('5.96 GB')
    expect(within(current).getByText('GPU 顯存').closest('tr')).toHaveTextContent('40%')
    const samples = screen.getAllByRole('table')[1]
    expect(within(samples).getByText('phoebe-embedder-1')).toBeInTheDocument()
    await userEvent.selectOptions(screen.getByRole('combobox', { name: '來源' }), 'GPU')
    await waitFor(() => expect(api.monitorSamples).toHaveBeenLastCalledWith(15, 'GPU'))
  })

  it('says when the monitor service is not sampling', async () => {
    vi.mocked(api.monitorStatus).mockResolvedValue({ collector: null, running: false, latest: [] })
    render(<MonitorPage />)
    await userEvent.click(screen.getByRole('tab', { name: '資源使用' }))
    expect(await screen.findByRole('status')).toHaveTextContent('monitor 服務沒有在取樣')
  })

  it('draws the charts and the task track for the chosen range', async () => {
    render(<MonitorPage />)
    await userEvent.click(screen.getByRole('tab', { name: '圖表' }))
    expect(await screen.findByRole('img', { name: '各容器的記憶體' })).toBeInTheDocument()
    // Phoebe services by name, other projects folded into one line
    const legend = screen.getByRole('list', { name: '各容器的記憶體的圖例' })
    expect(within(legend).getAllByRole('listitem').map((li) => li.textContent)).toEqual([
      'embedder',
      '其他容器',
    ])
    expect(screen.getByRole('img', { name: 'GPU 使用率' })).toBeInTheDocument()
    expect(screen.getByLabelText('文件處理：草莓園入園方案.pdf，完成，1 分 10 秒')).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: '6 小時' }))
    await waitFor(() => expect(api.monitorSeries).toHaveBeenLastCalledWith(360))
    expect(screen.getByRole('button', { name: '6 小時' })).toHaveAttribute('aria-pressed', 'true')
  })
})
