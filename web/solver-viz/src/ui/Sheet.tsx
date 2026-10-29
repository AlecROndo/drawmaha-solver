/**
 * The sheet the spot head's mix bar opens under it: the panes, one map per
 * action. Escape closes it, and so does the bar that opened it. It is a
 * sheet rather than a floating popover because three maps want the centre
 * column's width, and because a sheet cannot be clipped by the window's
 * edge.
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
