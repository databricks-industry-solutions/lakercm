import { useEffect, useState } from 'react'
import { useNavigate, useParams } from 'react-router'
import './SettingsPage.css'

const SECTIONS = [
  { key: 'profile', label: 'Profile' },
  { key: 'display', label: 'Display' },
]

const THEME_OPTIONS = [
  {
    key: 'light',
    label: 'Light',
    description: 'Bright background',
    icon: (
      <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="4" />
        <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41" />
      </svg>
    ),
  },
  {
    key: 'dark',
    label: 'Dark',
    description: 'Easier on the eyes',
    icon: (
      <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
        <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
      </svg>
    ),
  },
  {
    key: 'system',
    label: 'System',
    description: 'Follow OS setting',
    icon: (
      <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
        <rect x="2" y="3" width="20" height="14" rx="2" />
        <path d="M8 21h8M12 17v4" />
      </svg>
    ),
  },
]

const DENSITY_OPTIONS = [
  { key: 'relaxed', label: 'Relaxed', description: 'Generous spacing' },
  { key: 'compact', label: 'Compact', description: 'Tighter spacing' },
]

// Old localStorage value before the rename — silently migrate.
function normalizeDensity(value) {
  if (value === 'comfortable') return 'relaxed'
  if (value === 'compact' || value === 'relaxed') return value
  return 'relaxed'
}

function applyTheme(pref) {
  const resolved =
    pref === 'system'
      ? window.matchMedia('(prefers-color-scheme: dark)').matches
        ? 'dark'
        : 'light'
      : pref
  document.documentElement.dataset.theme = resolved
  document.documentElement.dataset.themePref = pref
}

function applyDensity(value) {
  document.documentElement.dataset.density = value
}

function SettingsPage({ currentUser }) {
  const navigate = useNavigate()
  const { section } = useParams()
  const activeSection = SECTIONS.find((s) => s.key === section)
    ? section
    : 'profile'

  const [theme, setTheme] = useState(
    () => localStorage.getItem('lakercm.theme') || 'light'
  )
  const [density, setDensity] = useState(() =>
    normalizeDensity(localStorage.getItem('lakercm.density'))
  )

  // Reflect "system" preference when the OS toggles light/dark.
  useEffect(() => {
    if (theme !== 'system') return
    const mq = window.matchMedia('(prefers-color-scheme: dark)')
    const onChange = () => applyTheme('system')
    mq.addEventListener('change', onChange)
    return () => mq.removeEventListener('change', onChange)
  }, [theme])

  const onSelectTheme = (key) => {
    setTheme(key)
    localStorage.setItem('lakercm.theme', key)
    applyTheme(key)
  }

  const onSelectDensity = (key) => {
    setDensity(key)
    localStorage.setItem('lakercm.density', key)
    applyDensity(key)
  }

  const email = currentUser?.email || '—'
  const displayName = currentUser?.display_name || email
  const initial = displayName ? displayName.charAt(0).toUpperCase() : '?'

  return (
    <div className="settings-page">
      <div className="settings-container">
        <header className="settings-page-header">
          <h1 className="settings-title">Settings</h1>
          <p className="settings-subtitle">Manage your account preferences</p>
        </header>

        <div className="settings-shell">
          <nav className="settings-rail" aria-label="Settings sections">
            {SECTIONS.map((s) => (
              <button
                key={s.key}
                type="button"
                className={`settings-rail-item ${
                  activeSection === s.key ? 'active' : ''
                }`}
                onClick={() => navigate(`/settings/${s.key}`)}
              >
                {s.label}
              </button>
            ))}
          </nav>

          <main className="settings-pane">
            {activeSection === 'profile' && (
              <section className="settings-section">
                <header className="settings-section-header">
                  <h2 className="settings-section-title">Profile</h2>
                  <p className="settings-section-subtitle">
                    Your account information from Databricks
                  </p>
                </header>

                <div className="settings-identity">
                  <div className="settings-identity-avatar">{initial}</div>
                  <div className="settings-identity-text">
                    <div className="settings-identity-name">{displayName}</div>
                    <div className="settings-identity-email">{email}</div>
                  </div>
                </div>

                <dl className="settings-fields">
                  <div className="settings-field">
                    <dt>Display name</dt>
                    <dd>{currentUser?.display_name || '—'}</dd>
                  </div>
                  <div className="settings-field">
                    <dt>Email</dt>
                    <dd>{email}</dd>
                  </div>
                  <div className="settings-field">
                    <dt>Identity source</dt>
                    <dd>Databricks workspace</dd>
                  </div>
                  <div className="settings-field">
                    <dt>Role</dt>
                    <dd>
                      {currentUser?.is_admin ? (
                        <span className="settings-pill admin">Admin</span>
                      ) : (
                        <span className="settings-pill">Reviewer</span>
                      )}
                    </dd>
                  </div>
                </dl>

                <div className="settings-info-banner">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                    <circle cx="12" cy="12" r="10" />
                    <path d="M12 16v-4M12 8h.01" />
                  </svg>
                  <span>
                    Profile fields are managed by your Databricks workspace
                    administrator. Update them via the Databricks admin
                    console.
                  </span>
                </div>
              </section>
            )}

            {activeSection === 'display' && (
              <section className="settings-section">
                <header className="settings-section-header">
                  <h2 className="settings-section-title">Display</h2>
                  <p className="settings-section-subtitle">
                    Adjust how LakeRCM looks on this device
                  </p>
                </header>

                <div className="settings-control">
                  <div className="settings-control-label">Theme</div>
                  <div className="theme-cards">
                    {THEME_OPTIONS.map((opt) => (
                      <button
                        key={opt.key}
                        type="button"
                        className={`theme-card ${
                          theme === opt.key ? 'selected' : ''
                        }`}
                        onClick={() => onSelectTheme(opt.key)}
                        aria-pressed={theme === opt.key}
                      >
                        <div className="theme-card-icon">{opt.icon}</div>
                        <div className="theme-card-label">{opt.label}</div>
                        <div className="theme-card-desc">{opt.description}</div>
                        {theme === opt.key && (
                          <div className="theme-card-check" aria-hidden="true">
                            <svg width="10" height="10" viewBox="0 0 12 12" fill="none">
                              <path d="M2.5 6.5l2.5 2.5 4.5-5" stroke="white" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
                            </svg>
                          </div>
                        )}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="settings-control">
                  <div className="settings-control-label">Density</div>
                  <div className="density-options">
                    {DENSITY_OPTIONS.map((opt) => (
                      <label
                        key={opt.key}
                        className={`density-option ${
                          density === opt.key ? 'selected' : ''
                        }`}
                      >
                        <input
                          type="radio"
                          name="density"
                          value={opt.key}
                          checked={density === opt.key}
                          onChange={() => onSelectDensity(opt.key)}
                        />
                        <div>
                          <div className="density-option-label">{opt.label}</div>
                          <div className="density-option-desc">
                            {opt.description}
                          </div>
                        </div>
                      </label>
                    ))}
                  </div>
                </div>

                <div className="settings-footnote">
                  Preferences are stored on this device only.
                </div>
              </section>
            )}
          </main>
        </div>
      </div>
    </div>
  )
}

export default SettingsPage
