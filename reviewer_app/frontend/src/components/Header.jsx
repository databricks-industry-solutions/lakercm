import './Header.css';
import UserMenu from './UserMenu';

function Header({ activeTab, onNavigate, currentUser }) {
  const email = currentUser?.email;

  return (
    <header className="modern-header">
      <div className="container">
        <div className="header-content">
          <div className="header-logo" onClick={() => onNavigate('overview')}>
            <svg width="36" height="36" viewBox="0 0 36 36" fill="none">
              <rect width="36" height="36" rx="8" fill="oklch(0.55 0.12 195)"/>
              <path d="M9 12.5h18M9 18h18M9 23.5h12" stroke="white" strokeWidth="2.5" strokeLinecap="round"/>
              <circle cx="27" cy="23.5" r="4.5" fill="#22c55e" stroke="white" strokeWidth="1.5"/>
              <path d="M25.2 23.5l1.2 1.2 2.4-2.4" stroke="white" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            <span className="logo-text">LakeRCM</span>
          </div>

          <nav className="header-nav">
            <button
              className={`nav-link ${activeTab === 'overview' ? 'active' : ''}`}
              onClick={() => onNavigate('overview')}
            >
              Overview
            </button>
            <button
              className={`nav-link ${activeTab === 'review' ? 'active' : ''}`}
              onClick={() => onNavigate('review')}
            >
              Reviewer
            </button>
            <button
              className={`nav-link ${activeTab === 'chat' ? 'active' : ''}`}
              onClick={() => onNavigate('chat')}
            >
              Assistant
            </button>
          </nav>

          <div className="header-actions">
            {email ? (
              <UserMenu currentUser={currentUser} />
            ) : (
              <span className="header-badge">LakeRCM</span>
            )}
          </div>
        </div>
      </div>
    </header>
  );
}

export default Header;
