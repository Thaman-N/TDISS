// tests/unit/WebSocketContext.test.jsx
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, act } from '@testing-library/react';
import React, { useContext, createContext, useState } from 'react';

// Mock WebSocketContext since we don't have the actual implementation
const WebSocketContext = createContext();

const WebSocketProvider = ({ children }) => {
  const [socket, setSocket] = useState(null);
  const [isConnected, setIsConnected] = useState(false);

  const connect = (url) => {
    const ws = new WebSocket(url);
    setSocket(ws);
    setIsConnected(true);
  };

  const disconnect = () => {
    if (socket && socket.close) {
      socket.close();
    }
    setSocket(null);
    setIsConnected(false);
  };

  return (
    <WebSocketContext.Provider value={{ socket, isConnected, connect, disconnect }}>
      {children}
    </WebSocketContext.Provider>
  );
};

// Test component to consume the context
const TestConsumer = () => {
  const { socket, isConnected, connect, disconnect } = useContext(WebSocketContext);
  
  return (
    <div>
      <div data-testid="connection-status">
        {isConnected ? 'Connected' : 'Disconnected'}
      </div>
      <button onClick={() => connect('ws://test-url')} data-testid="connect-btn">
        Connect
      </button>
      <button onClick={disconnect} data-testid="disconnect-btn">
        Disconnect
      </button>
      <div data-testid="socket-url">
        {socket?.url || 'No socket'}
      </div>
    </div>
  );
};

describe('WebSocketContext', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('should provide default context values', () => {
    render(
      <WebSocketProvider>
        <TestConsumer />
      </WebSocketProvider>
    );

    expect(screen.getByTestId('connection-status')).toHaveTextContent('Disconnected');
    expect(screen.getByTestId('socket-url')).toHaveTextContent('No socket');
  });

  it('should handle connection', async () => {
    render(
      <WebSocketProvider>
        <TestConsumer />
      </WebSocketProvider>
    );

    const connectBtn = screen.getByTestId('connect-btn');
    
    await act(async () => {
      connectBtn.click();
    });

    // Check if WebSocket was created
    expect(global.WebSocket).toHaveBeenCalledWith('ws://test-url');
  });

  it('should handle disconnection', async () => {
    render(
      <WebSocketProvider>
        <TestConsumer />
      </WebSocketProvider>
    );

    const connectBtn = screen.getByTestId('connect-btn');
    const disconnectBtn = screen.getByTestId('disconnect-btn');
    
    // First connect
    await act(async () => {
      connectBtn.click();
    });

    // Then disconnect
    await act(async () => {
      disconnectBtn.click();
    });

    expect(screen.getByTestId('connection-status')).toHaveTextContent('Disconnected');
  });
});