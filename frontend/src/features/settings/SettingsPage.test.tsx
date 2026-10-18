import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { HashRouter } from 'react-router'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { ApiError, api } from '../../api/client'
import type { LlmSettings } from '../../api/types'
import { NavGuardProvider, useNavGuard } from '../../shell/navGuard'
import SettingsPage from './SettingsPage'

vi.mock('../../api/client', async (orig) => ({
  ...(await orig<typeof import('../../api/client')>()),
  api: { getSettings: vi.fn(), saveSettings: vi.fn(), listModels: vi.fn() },
}))

function settings(over: Partial<LlmSettings> = {}): LlmSettings {
  return {
    llm_provider: 'ollama',
    ollama_model: 'qwen3',
    gemini_model: 'gemini-2.5-flash',
    system_prompt: '',
    temperature: null,
    providers: {
      ollama: { available: true, reason: '' },
      gemini: { available: true, reason: '' },
    },
    ...over,
  }
}

function Leave() {
  const { confirmLeave } = useNavGuard()
  return <button onClick={() => void confirmLeave()}>離開</button>
}

function setup() {
  return render(
    <HashRouter>
      <NavGuardProvider>
        <SettingsPage />
        <Leave />
      </NavGuardProvider>
    </HashRouter>,
  )
}

describe('SettingsPage', () => {
  beforeEach(() => {
    vi.resetAllMocks()
    vi.mocked(api.getSettings).mockResolvedValue(settings())
    vi.mocked(api.listModels).mockImplementation(async (p) =>
      p === 'ollama' ? ['qwen3', 'llama3'] : ['gemini-2.5-flash', 'gemini-2.5-pro'],
    )
  })

  it('loads settings and the model list for the provider', async () => {
    setup()
    const select = await screen.findByRole('combobox', { name: '模型' })
    await waitFor(() => expect(within(select).getAllByRole('option')).toHaveLength(2))
    expect(api.listModels).toHaveBeenCalledWith('ollama')
    expect(select).toHaveValue('qwen3')
    expect(screen.getByRole('radio', { name: /Ollama/ })).toBeChecked()
    expect(screen.getByText('0 / 4000')).toBeInTheDocument()
  })

  it('disables an unavailable provider with its reason', async () => {
    vi.mocked(api.getSettings).mockResolvedValue(
      settings({
        providers: {
          ollama: { available: true, reason: '' },
          gemini: { available: false, reason: '未設定 GEMINI_API_KEY' },
        },
      }),
    )
    setup()
    expect(await screen.findByRole('radio', { name: /Gemini/ })).toBeDisabled()
    expect(screen.getByText('未設定 GEMINI_API_KEY')).toBeInTheDocument()
  })

  it("switching provider fetches that provider's models", async () => {
    setup()
    await screen.findByRole('combobox', { name: '模型' })
    await userEvent.click(screen.getByRole('radio', { name: /Gemini/ }))
    await waitFor(() => expect(api.listModels).toHaveBeenCalledWith('gemini'))
    const select = screen.getByRole('combobox', { name: '模型' })
    await waitFor(() => expect(select).toHaveValue('gemini-2.5-flash'))
    expect(screen.getByRole('option', { name: 'gemini-2.5-pro' })).toBeInTheDocument()
  })

  it('shows saved model missing from list', async () => {
    vi.mocked(api.getSettings).mockResolvedValue(settings({ ollama_model: 'gone' }))
    setup()
    expect(
      await screen.findByRole('option', { name: 'gone（清單中找不到）' }),
    ).toBeInTheDocument()
    expect(screen.getByRole('combobox', { name: '模型' })).toHaveValue('gone')
  })

  it('model list failure shows reason and retry', async () => {
    vi.mocked(api.listModels).mockRejectedValueOnce(new ApiError(503, '無法連線到 Ollama'))
    setup()
    expect(await screen.findByText('無法連線到 Ollama')).toBeInTheDocument()
    expect(screen.getByRole('option', { name: 'qwen3' })).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: '重試' }))
    expect(await screen.findByRole('option', { name: 'llama3' })).toBeInTheDocument()
    expect(screen.queryByText('無法連線到 Ollama')).toBeNull()
    expect(api.listModels).toHaveBeenCalledTimes(2)
  })

  it('temperature uses default until unchecked', async () => {
    vi.mocked(api.saveSettings).mockImplementation(async (p) => settings(p))
    setup()
    const box = await screen.findByRole('checkbox', { name: '使用預設' })
    expect(box).toBeChecked()
    expect(screen.queryByRole('slider')).toBeNull()
    await userEvent.click(box)
    const slider = screen.getByRole('slider', { name: 'temperature' })
    fireEvent.change(slider, { target: { value: '1.4' } })
    expect(screen.getByText('1.4')).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: '儲存' }))
    await waitFor(() => expect(api.saveSettings).toHaveBeenCalledWith({ temperature: 1.4 }))
    await userEvent.click(screen.getByRole('checkbox', { name: '使用預設' }))
    await userEvent.click(screen.getByRole('button', { name: '儲存' }))
    await waitFor(() =>
      expect(api.saveSettings).toHaveBeenLastCalledWith({ temperature: null }),
    )
  })

  it('saves only changed fields and shows 已儲存', async () => {
    vi.mocked(api.saveSettings).mockImplementation(async (p) => settings(p))
    setup()
    const prompt = await screen.findByRole('textbox', { name: '系統提示詞' })
    await userEvent.type(prompt, 'hello')
    expect(screen.getByText('5 / 4000')).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: '儲存' }))
    expect(await screen.findByText('已儲存')).toBeInTheDocument()
    expect(api.saveSettings).toHaveBeenCalledWith({ system_prompt: 'hello' })
    expect(screen.getByRole('button', { name: '儲存' })).toBeDisabled()
  })

  it('shows server error on failure', async () => {
    vi.mocked(api.saveSettings).mockRejectedValue(
      new ApiError(409, 'Gemini 目前不可用：未設定 GEMINI_API_KEY'),
    )
    setup()
    await userEvent.type(await screen.findByRole('textbox', { name: '系統提示詞' }), 'x')
    await userEvent.click(screen.getByRole('button', { name: '儲存' }))
    expect(
      await screen.findByText('Gemini 目前不可用：未設定 GEMINI_API_KEY'),
    ).toBeInTheDocument()
    expect(screen.queryByText('已儲存')).toBeNull()
  })

  it('blocks saving a prompt over 4000 characters', async () => {
    setup()
    const prompt = await screen.findByRole('textbox', { name: '系統提示詞' })
    fireEvent.change(prompt, { target: { value: 'a'.repeat(4001) } })
    expect(screen.getByText('4001 / 4000')).toBeInTheDocument()
    expect(screen.getByText('系統提示詞不可超過 4000 字')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '儲存' })).toBeDisabled()
  })

  it('asks before leaving with unsaved changes', async () => {
    setup()
    await userEvent.click(screen.getByRole('button', { name: '離開' }))
    expect(screen.queryByText('有未儲存的變更，確定要離開？')).toBeNull()
    await userEvent.type(await screen.findByRole('textbox', { name: '系統提示詞' }), 'x')
    await userEvent.click(screen.getByRole('button', { name: '離開' }))
    expect(await screen.findByText('有未儲存的變更，確定要離開？')).toBeInTheDocument()
  })

  it('does not ask after saving', async () => {
    vi.mocked(api.saveSettings).mockImplementation(async (p) => settings(p))
    setup()
    await userEvent.type(await screen.findByRole('textbox', { name: '系統提示詞' }), 'x')
    await userEvent.click(screen.getByRole('button', { name: '儲存' }))
    await screen.findByText('已儲存')
    await userEvent.click(screen.getByRole('button', { name: '離開' }))
    expect(screen.queryByText('有未儲存的變更，確定要離開？')).toBeNull()
  })

  it('shows load failure with retry', async () => {
    vi.mocked(api.getSettings).mockRejectedValueOnce(new ApiError(403, '需要管理員權限'))
    setup()
    expect(await screen.findByText('需要管理員權限')).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: '重試' }))
    expect(await screen.findByRole('combobox', { name: '模型' })).toBeInTheDocument()
  })
})
