import { useEffect, useMemo, useRef, useState, useCallback } from 'react'
import { useNavigate } from 'react-router'
import './UserMenu.css'

function UserMenu({ currentUser }) {
  const [open, setOpen] = useState(false)
  const [focusIndex, setFocusIndex] = useState(0)
  const wrapperRef = useRef(null)
  const itemRefs = useRef([])
  const navigate = useNavigate()

  const email = currentUser?.email
  const displayName = currentUser?.display_name || email
  const initial = displayName ? displayName.charAt(0).toUpperCase() : '?'
  const isAdmin = Boolean(currentUser?.is_admin)

  // Memoized on isAdmin: this array is a dependency of the listener effect
  // below, so rebuilding it every render re-subscribed the document-level
  // mousedown/keydown handlers every render.
  const items = useMemo(
    () => [
      { key: 'settings', label: 'Settings', path: '/settings' },
      ...(isAdmin
        ? [{ key: 'admin', label: 'Admin', path: '/admin', accent: true }]
        : []),
    ],
    [isAdmin]
  )

  const close = useCallback(() => {
    setOpen(false)
    setFocusIndex(0)
  }, [])

  useEffect(() => {
    if (!open) return
    const onMouseDown = (e) => {
      if (wrapperRef.current && !wrapperRef.current.contains(e.target)) close()
    }
    const onKey = (e) => {
      if (e.key === 'Escape') {
        e.preventDefault()
        close()
      } else if (e.key === 'ArrowDown') {
        e.preventDefault()
        setFocusIndex((i) => Math.min(items.length - 1, i + 1))
      } else if (e.key === 'ArrowUp') {
        e.preventDefault()
        setFocusIndex((i) => Math.max(0, i - 1))
      } else if (e.key === 'Enter' || e.key === ' ') {
        const item = items[focusIndex]
        if (item) {
          e.preventDefault()
          navigate(item.path)
          close()
        }
      }
    }
    document.addEventListener('mousedown', onMouseDown)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onMouseDown)
      document.removeEventListener('keydown', onKey)
    }
  }, [open, focusIndex, items, navigate, close])

  useEffect(() => {
    if (open) itemRefs.current[focusIndex]?.focus()
  }, [open, focusIndex])

  if (!email) return null

  return (
    <div className="user-menu" ref={wrapperRef}>
      <button
        type="button"
        className="user-menu-trigger"
        aria-haspopup="menu"
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
      >
        <div className="user-menu-avatar">{initial}</div>
        <span className="user-menu-email">{displayName}</span>
        <svg
          className={`user-menu-chevron ${open ? 'open' : ''}`}
          width="12"
          height="12"
          viewBox="0 0 12 12"
          fill="none"
          aria-hidden="true"
        >
          <path d="M3 4.5l3 3 3-3" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </button>

      {open && (
        <div className="user-menu-panel" role="menu">
          <div className="user-menu-header">
            <div className="user-menu-header-name">{displayName}</div>
            {displayName !== email && (
              <div className="user-menu-header-email">{email}</div>
            )}
          </div>
          <div className="user-menu-divider" />
          {items.map((item, idx) => (
            <button
              key={item.key}
              ref={(el) => (itemRefs.current[idx] = el)}
              type="button"
              className={`user-menu-item ${item.accent ? 'accent' : ''}`}
              role="menuitem"
              tabIndex={focusIndex === idx ? 0 : -1}
              onClick={() => {
                navigate(item.path)
                close()
              }}
              onMouseEnter={() => setFocusIndex(idx)}
            >
              {item.label}
              {item.accent && <span className="user-menu-badge">admin</span>}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

export default UserMenu
