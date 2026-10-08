/**
 * The sheet a header control opens under the spot header: the three panes,
 * the deck, or nothing. One sheet at a time; Escape closes it, and so does
 * the control that opened it. It is a sheet rather than a floating popover
 * because its contents (a 13×4 deck, three maps) want the window's width,
 * and because a sheet cannot be clipped by the window's edge.
 */

import { useEffect, useRef } from 'react'

export function Sheet({
  title,
  onClose,
  children,
}: {
  title: string
  onClose: () => void
  children: React.ReactNode
}) {
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    document.addEventListener('keydown', onKey)
    // Focus lands on the sheet so a keyboard user is where the content is.
    ref.current?.focus()
    return () => document.removeEventListener('keydown', onKey)
  }, [onClose])
  return (
    <div className="sheet" role="dialog" aria-label={title} tabIndex={-1} ref={ref}>
      <div className="sheet-bar">
        <span className="lbl">{title}</span>
        <button className="btn ghost" type="button" onClick={onClose}>
          Close
        </button>
      </div>
      {children}
    </div>
  )
}
