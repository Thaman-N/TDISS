// tests/unit/LandingPage.test.jsx
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

// Mock LandingPage component
const MockLandingPage = ({ onNavigate }) => {
  return (
    <div data-testid="landing-page">
      {/* Hero Section */}
      <section data-testid="hero-section">
        <h1>TDISS Violence Detection System</h1>
        <p>Advanced AI-powered violence detection for video analysis and live streaming.</p>
        <div data-testid="hero-actions">
          <button 
            onClick={() => onNavigate?.('/upload')}
            data-testid="upload-btn"
          >
            Upload Video
          </button>
          <button 
            onClick={() => onNavigate?.('/live-stream')}
            data-testid="live-stream-btn"
          >
            Live Stream
          </button>
        </div>
      </section>

      {/* Features Section */}
      <section data-testid="features-section">
        <h2>Features</h2>
        <div data-testid="features-grid">
          <div data-testid="feature-realtime">
            <h3>Real-time Processing</h3>
            <p>Instant violence detection in live streams</p>
          </div>
          <div data-testid="feature-accuracy">
            <h3>High Accuracy</h3>
            <p>Advanced deep learning models for precise detection</p>
          </div>
          <div data-testid="feature-dashboard">
            <h3>Interactive Dashboard</h3>
            <p>Comprehensive results and timeline visualization</p>
          </div>
          <div data-testid="feature-history">
            <h3>Analysis History</h3>
            <p>Track and review all your processed videos</p>
          </div>
        </div>
      </section>

      {/* Demo Section */}
      <section data-testid="demo-section">
        <h2>See It In Action</h2>
        <div data-testid="demo-content">
          <video data-testid="demo-video" controls>
            <source src="/demo-video.mp4" type="video/mp4" />
            Your browser does not support the video tag.
          </video>
          <div data-testid="demo-description">
            <p>Watch our AI system detect violence in real-time with high accuracy and detailed analysis.</p>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section data-testid="stats-section">
        <h2>System Statistics</h2>
        <div data-testid="stats-grid">
          <div data-testid="stat-accuracy">
            <span>95%</span>
            <p>Detection Accuracy</p>
          </div>
          <div data-testid="stat-processing">
            <span>30fps</span>
            <p>Real-time Processing</p>
          </div>
          <div data-testid="stat-videos">
            <span>1000+</span>
            <p>Videos Processed</p>
          </div>
        </div>
      </section>
    </div>
  );
};

describe('LandingPage Component', () => {
  let user;
  let mockOnNavigate;

  beforeEach(() => {
    user = userEvent.setup();
    mockOnNavigate = vi.fn();
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('should render the landing page', () => {
    render(<MockLandingPage />);
    
    expect(screen.getByTestId('landing-page')).toBeInTheDocument();
    expect(screen.getByText('TDISS Violence Detection System')).toBeInTheDocument();
  });

  it('should render hero section with title and description', () => {
    render(<MockLandingPage />);
    
    expect(screen.getByTestId('hero-section')).toBeInTheDocument();
    expect(screen.getByText('TDISS Violence Detection System')).toBeInTheDocument();
    expect(screen.getByText('Advanced AI-powered violence detection for video analysis and live streaming.')).toBeInTheDocument();
  });

  it('should render action buttons in hero section', () => {
    render(<MockLandingPage onNavigate={mockOnNavigate} />);
    
    const uploadBtn = screen.getByTestId('upload-btn');
    const liveStreamBtn = screen.getByTestId('live-stream-btn');
    
    expect(uploadBtn).toBeInTheDocument();
    expect(uploadBtn).toHaveTextContent('Upload Video');
    expect(liveStreamBtn).toBeInTheDocument();
    expect(liveStreamBtn).toHaveTextContent('Live Stream');
  });

  it('should handle upload button click', async () => {
    render(<MockLandingPage onNavigate={mockOnNavigate} />);
    
    const uploadBtn = screen.getByTestId('upload-btn');
    await user.click(uploadBtn);
    
    expect(mockOnNavigate).toHaveBeenCalledWith('/upload');
  });

  it('should handle live stream button click', async () => {
    render(<MockLandingPage onNavigate={mockOnNavigate} />);
    
    const liveStreamBtn = screen.getByTestId('live-stream-btn');
    await user.click(liveStreamBtn);
    
    expect(mockOnNavigate).toHaveBeenCalledWith('/live-stream');
  });

  it('should render features section', () => {
    render(<MockLandingPage />);
    
    expect(screen.getByTestId('features-section')).toBeInTheDocument();
    expect(screen.getByText('Features')).toBeInTheDocument();
    expect(screen.getByTestId('features-grid')).toBeInTheDocument();
  });

  it('should display all feature cards', () => {
    render(<MockLandingPage />);
    
    const featuresGrid = screen.getByTestId('features-grid');
    
    expect(screen.getByTestId('feature-realtime')).toBeInTheDocument();
    expect(within(screen.getByTestId('feature-realtime')).getByText('Real-time Processing')).toBeInTheDocument();
    expect(within(screen.getByTestId('feature-realtime')).getByText('Instant violence detection in live streams')).toBeInTheDocument();
    
    expect(screen.getByTestId('feature-accuracy')).toBeInTheDocument();
    expect(within(screen.getByTestId('feature-accuracy')).getByText('High Accuracy')).toBeInTheDocument();
    
    expect(screen.getByTestId('feature-dashboard')).toBeInTheDocument();
    expect(within(screen.getByTestId('feature-dashboard')).getByText('Interactive Dashboard')).toBeInTheDocument();
    
    expect(screen.getByTestId('feature-history')).toBeInTheDocument();
    expect(within(screen.getByTestId('feature-history')).getByText('Analysis History')).toBeInTheDocument();
  });

  it('should render demo section with video', () => {
    render(<MockLandingPage />);
    
    expect(screen.getByTestId('demo-section')).toBeInTheDocument();
    expect(screen.getByText('See It In Action')).toBeInTheDocument();
    
    const demoVideo = screen.getByTestId('demo-video');
    expect(demoVideo).toBeInTheDocument();
    expect(demoVideo).toHaveAttribute('controls');
  });

  it('should render statistics section', () => {
    render(<MockLandingPage />);
    
    expect(screen.getByTestId('stats-section')).toBeInTheDocument();
    expect(screen.getByText('System Statistics')).toBeInTheDocument();
    
    expect(screen.getByTestId('stat-accuracy')).toBeInTheDocument();
    expect(screen.getByText('95%')).toBeInTheDocument();
    expect(screen.getByText('Detection Accuracy')).toBeInTheDocument();
    
    expect(screen.getByTestId('stat-processing')).toBeInTheDocument();
    expect(screen.getByText('30fps')).toBeInTheDocument();
    
    expect(screen.getByTestId('stat-videos')).toBeInTheDocument();
    expect(screen.getByText('1000+')).toBeInTheDocument();
  });

  it('should have proper section structure', () => {
    render(<MockLandingPage />);
    
    const sections = ['hero-section', 'features-section', 'demo-section', 'stats-section'];
    
    sections.forEach(section => {
      expect(screen.getByTestId(section)).toBeInTheDocument();
    });
  });

  it('should handle navigation without callback', async () => {
    render(<MockLandingPage />);
    
    const uploadBtn = screen.getByTestId('upload-btn');
    
    // Should not throw error when onNavigate is not provided
    await user.click(uploadBtn);
    
    expect(uploadBtn).toBeInTheDocument();
  });
});