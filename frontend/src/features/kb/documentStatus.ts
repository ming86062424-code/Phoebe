import type { KbDocument } from '../../api/types'

/** How a document's status reads in the UI. */
export const STATUS_TEXT: Record<KbDocument['status'], string> = {
  pending: '排隊中',
  queued: '排隊中',
  parsing: '解析中',
  indexing: '建立索引中',
  ready: '完成',
  failed: '失敗',
}
