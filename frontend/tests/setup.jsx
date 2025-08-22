// tests/setup.jsx
import '@testing-library/jest-dom'
import { vi } from 'vitest'
import React from 'react'

// Mock IntersectionObserver
global.IntersectionObserver = vi.fn().mockImplementation((callback, options) => ({
  observe: vi.fn(),
  unobserve: vi.fn(),
  disconnect: vi.fn(),
  root: null,
  rootMargin: '',
  thresholds: []
}))

// Mock ResizeObserver
global.ResizeObserver = vi.fn().mockImplementation(() => ({
  observe: vi.fn(),
  unobserve: vi.fn(),
  disconnect: vi.fn(),
}))

// Mock WebSocket with proper close function
const mockWebSocket = vi.fn().mockImplementation((url) => {
  const ws = {
    url,
    send: vi.fn(),
    close: vi.fn(), // Make sure close is a function
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    readyState: 1, // OPEN
    CONNECTING: 0,
    OPEN: 1,
    CLOSING: 2,
    CLOSED: 3,
  }
  
  // Simulate connection
  setTimeout(() => {
    if (ws.addEventListener.mock.calls.length > 0) {
      const openHandler = ws.addEventListener.mock.calls.find(call => call[0] === 'open')
      if (openHandler && openHandler[1]) {
        openHandler[1]()
      }
    }
  }, 0)
  
  return ws
})

global.WebSocket = mockWebSocket
global.WebSocket.CONNECTING = 0
global.WebSocket.OPEN = 1
global.WebSocket.CLOSING = 2
global.WebSocket.CLOSED = 3

// Mock matchMedia
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: vi.fn().mockImplementation(query => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: vi.fn(),
    removeListener: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })),
})

// Mock react-router-dom with proper JSX
vi.mock('react-router-dom', async (importOriginal) => {
  const actual = await importOriginal()
  
  return {
    ...actual,
    useNavigate: () => vi.fn(),
    useParams: () => ({ jobId: 'test-job-id', streamId: 'test-stream-id' }),
    useSearchParams: () => [new URLSearchParams(), vi.fn()],
    Link: ({ children, to, ...props }) => <a href={to} {...props}>{children}</a>,
    BrowserRouter: ({ children }) => <div data-testid="browser-router">{children}</div>,
  }
})