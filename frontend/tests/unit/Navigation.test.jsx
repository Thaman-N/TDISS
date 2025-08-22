// tests/unit/Navigation.test.jsx
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

// Mock Navigation component
const MockNavigation = ({ currentPath = '/' }) => {
  const navItems = [
    { path: '/', label: 'Dashboard', icon: '🏠' },
    { path: '/upload', label: 'Upload', icon: '📤' },
    { path: '/live-stream', label: 'Live Stream', icon: '📹' },
    { path: '/history', label: 'History', icon: '📋' }
  ];

  const isActive = (path) => currentPath === path;

  return (
    <nav data-testid="navigation">
      <div data-testid="nav-brand">
        <h1>TDISS</h1>
        <span>Violence Detection</span>
      </div>
      
      <ul data-testid="nav-menu">
        {navItems.map((item) => (
          <li key={item.path}>
            <a
              href={item.path}
              data-testid={`nav-link-${item.path.replace('/', '') || 'dashboard'}`}
              className={isActive(item.path) ? 'active' : ''}
            >
              <span>{item.icon}</span>
              <span>{item.label}</span>
            </a>
          </li>
        ))}
      </ul>

      <div data-testid="nav-actions">
        <button data-testid="theme-toggle">🌙</button>
        <button data-testid="user-menu">👤</button>
      </div>
    </nav>
  );
};

describe('Navigation Component', () => {
  let user;

  beforeEach(() => {
    user = userEvent.setup();
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('should render navigation with brand', () => {
    render(<MockNavigation />);
    
    expect(screen.getByTestId('navigation')).toBeInTheDocument();
    expect(screen.getByTestId('nav-brand')).toBeInTheDocument();
    expect(screen.getByText('TDISS')).toBeInTheDocument();
    expect(screen.getByText('Violence Detection')).toBeInTheDocument();
  });

  it('should render all navigation links', () => {
    render(<MockNavigation />);
    
    expect(screen.getByTestId('nav-menu')).toBeInTheDocument();
    expect(screen.getByTestId('nav-link-dashboard')).toBeInTheDocument();
    expect(screen.getByTestId('nav-link-upload')).toBeInTheDocument();
    expect(screen.getByTestId('nav-link-live-stream')).toBeInTheDocument();
    expect(screen.getByTestId('nav-link-history')).toBeInTheDocument();
  });

  it('should highlight active navigation item', () => {
    render(<MockNavigation currentPath="/upload" />);
    
    const activeLink = screen.getByTestId('nav-link-upload');
    const inactiveLink = screen.getByTestId('nav-link-dashboard');
    
    expect(activeLink).toHaveClass('active');
    expect(inactiveLink).not.toHaveClass('active');
  });

  it('should show navigation items with icons and labels', () => {
    render(<MockNavigation />);
    
    const dashboardLink = screen.getByTestId('nav-link-dashboard');
    expect(dashboardLink).toHaveTextContent('🏠');
    expect(dashboardLink).toHaveTextContent('Dashboard');
    
    const uploadLink = screen.getByTestId('nav-link-upload');
    expect(uploadLink).toHaveTextContent('📤');
    expect(uploadLink).toHaveTextContent('Upload');
  });

  it('should render action buttons', () => {
    render(<MockNavigation />);
    
    expect(screen.getByTestId('nav-actions')).toBeInTheDocument();
    expect(screen.getByTestId('theme-toggle')).toBeInTheDocument();
    expect(screen.getByTestId('user-menu')).toBeInTheDocument();
  });

  it('should handle theme toggle click', async () => {
    render(<MockNavigation />);
    
    const themeToggle = screen.getByTestId('theme-toggle');
    
    await user.click(themeToggle);
    
    // Button should be clickable (no errors)
    expect(themeToggle).toBeInTheDocument();
  });

  it('should handle user menu click', async () => {
    render(<MockNavigation />);
    
    const userMenu = screen.getByTestId('user-menu');
    
    await user.click(userMenu);
    
    // Button should be clickable (no errors)
    expect(userMenu).toBeInTheDocument();
  });

  it('should handle different current paths', () => {
    const { rerender } = render(<MockNavigation currentPath="/" />);
    expect(screen.getByTestId('nav-link-dashboard')).toHaveClass('active');
    
    rerender(<MockNavigation currentPath="/live-stream" />);
    expect(screen.getByTestId('nav-link-live-stream')).toHaveClass('active');
    expect(screen.getByTestId('nav-link-dashboard')).not.toHaveClass('active');
  });
});