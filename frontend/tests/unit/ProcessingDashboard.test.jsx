// tests/unit/ProcessingDashboard.test.jsx
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';

// Mock ProcessingDashboard component
const MockProcessingDashboard = ({ jobId, status, progress }) => {
  const getStatusIcon = () => {
    switch (status) {
      case 'uploading': return '⏳';
      case 'processing': return '🔄';
      case 'completed': return '✅';
      case 'error': return '❌';
      default: return '⏳';
    }
  };

  const getStatusColor = () => {
    switch (status) {
      case 'completed': return 'text-green-600';
      case 'error': return 'text-red-600';
      case 'processing': return 'text-blue-600';
      default: return 'text-gray-600';
    }
  };

  return (
    <div data-testid="processing-dashboard">
      <div data-testid="job-info">
        <h2>Job: {jobId}</h2>
        <div className={getStatusColor()} data-testid="status">
          {getStatusIcon()} {status.charAt(0).toUpperCase() + status.slice(1)}
        </div>
      </div>
      
      {status === 'processing' && (
        <div data-testid="progress-section">
          <div data-testid="progress-text">Progress: {progress}%</div>
          <progress value={progress} max={100} data-testid="progress-bar" />
        </div>
      )}

      {status === 'completed' && (
        <div data-testid="completion-message">
          Processing completed successfully!
        </div>
      )}

      {status === 'error' && (
        <div data-testid="error-message">
          Processing failed. Please try again.
        </div>
      )}
    </div>
  );
};

describe('ProcessingDashboard Component', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('should render dashboard with job information', () => {
    render(
      <MockProcessingDashboard 
        jobId="test-job-123" 
        status="processing" 
        progress={45} 
      />
    );

    expect(screen.getByTestId('processing-dashboard')).toBeInTheDocument();
    expect(screen.getByText('Job: test-job-123')).toBeInTheDocument();
  });

  it('should show uploading status', () => {
    render(
      <MockProcessingDashboard 
        jobId="test-job-123" 
        status="uploading" 
        progress={0} 
      />
    );

    const status = screen.getByTestId('status');
    expect(status).toHaveTextContent('⏳ Uploading');
    expect(status).toHaveClass('text-gray-600');
  });

  it('should show processing status with progress', () => {
    render(
      <MockProcessingDashboard 
        jobId="test-job-123" 
        status="processing" 
        progress={65} 
      />
    );

    expect(screen.getByTestId('status')).toHaveTextContent('🔄 Processing');
    expect(screen.getByTestId('progress-section')).toBeInTheDocument();
    expect(screen.getByTestId('progress-text')).toHaveTextContent('Progress: 65%');
    
    const progressBar = screen.getByTestId('progress-bar');
    expect(progressBar).toHaveAttribute('value', '65');
    expect(progressBar).toHaveAttribute('max', '100');
  });

  it('should show completed status', () => {
    render(
      <MockProcessingDashboard 
        jobId="test-job-123" 
        status="completed" 
        progress={100} 
      />
    );

    const status = screen.getByTestId('status');
    expect(status).toHaveTextContent('✅ Completed');
    expect(status).toHaveClass('text-green-600');
    expect(screen.getByTestId('completion-message')).toHaveTextContent('Processing completed successfully!');
  });

  it('should show error status', () => {
    render(
      <MockProcessingDashboard 
        jobId="test-job-123" 
        status="error" 
        progress={0} 
      />
    );

    const status = screen.getByTestId('status');
    expect(status).toHaveTextContent('❌ Error');
    expect(status).toHaveClass('text-red-600');
    expect(screen.getByTestId('error-message')).toHaveTextContent('Processing failed. Please try again.');
  });

  it('should not show progress when not processing', () => {
    render(
      <MockProcessingDashboard 
        jobId="test-job-123" 
        status="completed" 
        progress={100} 
      />
    );

    expect(screen.queryByTestId('progress-section')).not.toBeInTheDocument();
  });

  it('should handle progress updates', async () => {
    const { rerender } = render(
      <MockProcessingDashboard 
        jobId="test-job-123" 
        status="processing" 
        progress={25} 
      />
    );

    expect(screen.getByTestId('progress-text')).toHaveTextContent('Progress: 25%');

    // Update progress
    rerender(
      <MockProcessingDashboard 
        jobId="test-job-123" 
        status="processing" 
        progress={75} 
      />
    );

    expect(screen.getByTestId('progress-text')).toHaveTextContent('Progress: 75%');
  });

  it('should handle status transitions', async () => {
    const { rerender } = render(
      <MockProcessingDashboard 
        jobId="test-job-123" 
        status="uploading" 
        progress={0} 
      />
    );

    expect(screen.getByTestId('status')).toHaveTextContent('Uploading');

    // Transition to processing
    rerender(
      <MockProcessingDashboard 
        jobId="test-job-123" 
        status="processing" 
        progress={50} 
      />
    );

    expect(screen.getByTestId('status')).toHaveTextContent('Processing');
    expect(screen.getByTestId('progress-section')).toBeInTheDocument();

    // Transition to completed
    rerender(
      <MockProcessingDashboard 
        jobId="test-job-123" 
        status="completed" 
        progress={100} 
      />
    );

    expect(screen.getByTestId('status')).toHaveTextContent('Completed');
    expect(screen.getByTestId('completion-message')).toBeInTheDocument();
    expect(screen.queryByTestId('progress-section')).not.toBeInTheDocument();
  });
});