import { useState, useEffect, useCallback } from 'react'
import ChatInterface from '../components/ChatInterface/ChatInterface'
import { getConversations, deleteConversation } from '../api/chatApi'
import './ChatPage.css'

// ChatPage: the sidebar/conversation-rail shell around ChatInterface, whose
// transport is AG-UI (speaking to the bare ag_ui_langgraph mount at
// /api/copilotkit). Conversation history + feedback CRUD come from
// reviewer_app's /api endpoints.

const LS_RAIL_COLLAPSED = 'lakercm:chat-rail-collapsed'

function ChatPage({ currentUser }) {
  const [conversations, setConversations] = useState([])
  const [activeId, setActiveId] = useState(() => crypto.randomUUID())
  const [streamingConvId, setStreamingConvId] = useState(null)
  const [collapsed, setCollapsed] = useState(() => {
    try {
      return window.localStorage.getItem(LS_RAIL_COLLAPSED) !== 'false'
    } catch {
      return true
    }
  })

  useEffect(() => {
    try {
      window.localStorage.setItem(LS_RAIL_COLLAPSED, String(collapsed))
    } catch {
      // ignore
    }
  }, [collapsed])

  const loadConversations = useCallback(async () => {
    try {
      const data = await getConversations()
      setConversations(data)
    } catch {
      // ignore
    }
  }, [])

  useEffect(() => {
    loadConversations()
  }, [loadConversations])

  const handleNewChat = () => {
    setActiveId(crypto.randomUUID())
  }

  const handleDelete = async (convId, e) => {
    e.stopPropagation()
    if (convId === streamingConvId) return
    try {
      await deleteConversation(convId)
      if (activeId === convId) setActiveId(null)
      await loadConversations()
    } catch {
      // ignore
    }
  }

  const handleSelect = (convId) => {
    setActiveId(convId)
  }

  return (
    <div className="chat-page">
      <aside
        className={`chat-page__sidebar ${
          collapsed ? 'chat-page__sidebar--collapsed' : ''
        }`}
      >
        <div className="chat-page__sidebar-header">
          <button
            className="chat-page__sidebar-toggle"
            onClick={() => setCollapsed((v) => !v)}
            title={collapsed ? 'Show conversations' : 'Hide conversations'}
            aria-label={collapsed ? 'Show conversations' : 'Hide conversations'}
          >
            <svg
              className={`chat-page__sidebar-toggle-icon ${
                collapsed ? '' : 'chat-page__sidebar-toggle-icon--open'
              }`}
              width="14"
              height="14"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <polyline points="9 6 15 12 9 18" />
            </svg>
          </button>
          {!collapsed && (
            <span className="chat-page__sidebar-title">Conversations</span>
          )}
        </div>
        <button
          className="chat-page__new-btn"
          onClick={handleNewChat}
          title="New chat"
          aria-label="New chat"
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
            <line x1="12" y1="5" x2="12" y2="19" />
            <line x1="5" y1="12" x2="19" y2="12" />
          </svg>
          {!collapsed && <span className="chat-page__new-btn-label">New chat</span>}
        </button>
        {!collapsed && (
          <div className="chat-page__conv-list">
            {conversations.length === 0 ? (
              <p className="chat-page__empty">No conversations yet</p>
            ) : (
              conversations.map((c) => (
                <div
                  key={c.id}
                  className={`chat-page__conv-item ${
                    activeId === c.id ? 'chat-page__conv-item--active' : ''
                  }`}
                  onClick={() => handleSelect(c.id)}
                >
                  <div className="chat-page__conv-info">
                    <span className="chat-page__conv-title">
                      {c.title || 'Untitled'}
                    </span>
                    <span className="chat-page__conv-meta">
                      {c.message_count} messages
                    </span>
                  </div>
                  <button
                    className="chat-page__conv-delete"
                    onClick={(e) => handleDelete(c.id, e)}
                    disabled={c.id === streamingConvId}
                    title={
                      c.id === streamingConvId
                        ? 'Finish the current response before deleting.'
                        : 'Delete'
                    }
                  >
                    &times;
                  </button>
                </div>
              ))
            )}
          </div>
        )}
      </aside>

      <div className="chat-page__main">
        <ChatInterface
          conversationId={activeId}
          currentUser={currentUser}
          onStreamingChange={setStreamingConvId}
        />
      </div>
    </div>
  )
}

export default ChatPage
