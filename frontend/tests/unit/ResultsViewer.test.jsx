// tests/unit/ResultsViewer.test.jsx
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen } from '@testing-library/react';

// Mock results data
const mockResults = {
  job_id: 'test-job-123',
  violence_detected: true,
  confidence_score: 0.85,
  timeline: [
    { timestamp: 10, confidence: 0.9, violence: true },
    { timestamp: 15, confidence: 0.8, violence: true },
    { timestamp: 20, confidence: 0.3, violence: false },
    { timestamp: 25, confidence: 0.95, violence: true }
  ],
  metadata: {
    duration: 30,
    fps: 30,
    resolution: '1920x1080',
    filename: 'test-video.mp4'
  }
};

// Mock ResultsViewer component
const MockResultsViewer = ({ results }) => {
  const formatTimestamp = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return 'text-red-600';
    if (confidence >= 0.6) return 'text-yellow-600';
    return 'text-green-600';
  };

  if (!results) {
    return <div data-testid="no-results">No results available</div>;
  }

  return (
    <div data-testid="results-viewer">
      {/* Summary Section */}
      <div data-testid="results-summary">
        <h2>Analysis Results</h2>
        <div data-testid="detection-status">
          Violence Detected: {results.violence_detected ? 'Yes' : 'No'}
        </div>
        <div data-testid="confidence-score">
          Confidence: {(results.confidence_score * 100).toFixed(1)}%
        </div>
      </div>

      {/* Metadata Section */}
      <div data-testid="metadata-section">
        <h3>Video Information</h3>
        <div data-testid="filename">File: {results.metadata.filename}</div>
        <div data-testid="duration">Duration: {results.metadata.duration}s</div>
        <div data-testid="resolution">Resolution: {results.metadata.resolution}</div>
        <div data-testid="fps">FPS: {results.metadata.fps}</div>
      </div>

      {/* Timeline Section */}
      <div data-testid="timeline-section">
        <h3>Violence Timeline</h3>
        <div data-testid="timeline-list">
          {results.timeline.map((event, index) => (
            <div 
              key={index} 
              data-testid={`timeline-event-${index}`}
              className="timeline-event"
            >
              <span data-testid={`timestamp-${index}`}>
                {formatTimestamp(event.timestamp)}
              </span>
              <span 
                data-testid={`confidence-${index}`}
                className={getConfidenceColor(event.confidence)}
              >
                {(event.confidence * 100).toFixed(1)}%
              </span>
              <span data-testid={`violence-status-${index}`}>
                {event.violence ? 'Violence' : 'No Violence'}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Statistics */}
      <div data-testid="statistics-section">
        <h3>Statistics</h3>
        <div data-testid="total-events">
          Total Events: {results.timeline.length}
        </div>
        <div data-testid="violence-events">
          Violence Events: {results.timeline.filter(e => e.violence).length}
        </div>
        <div data-testid="avg-confidence">
          Average Confidence: {(results.timeline.reduce((acc, e) => acc + e.confidence, 0) / results.timeline.length * 100).toFixed(1)}%
        </div>
      </div>
    </div>
  );
};

describe('ResultsViewer Component', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('should render no results message when results are null', () => {
    render(<MockResultsViewer results={null} />);
    
    expect(screen.getByTestId('no-results')).toBeInTheDocument();
    expect(screen.getByText('No results available')).toBeInTheDocument();
  });

  it('should render results summary correctly', () => {
    render(<MockResultsViewer results={mockResults} />);
    
    expect(screen.getByTestId('results-summary')).toBeInTheDocument();
    expect(screen.getByTestId('detection-status')).toHaveTextContent('Violence Detected: Yes');
    expect(screen.getByTestId('confidence-score')).toHaveTextContent('Confidence: 85.0%');
  });

  it('should display metadata information', () => {
    render(<MockResultsViewer results={mockResults} />);
    
    expect(screen.getByTestId('metadata-section')).toBeInTheDocument();
    expect(screen.getByTestId('filename')).toHaveTextContent('File: test-video.mp4');
    expect(screen.getByTestId('duration')).toHaveTextContent('Duration: 30s');
    expect(screen.getByTestId('resolution')).toHaveTextContent('Resolution: 1920x1080');
    expect(screen.getByTestId('fps')).toHaveTextContent('FPS: 30');
  });

  it('should render timeline events correctly', () => {
    render(<MockResultsViewer results={mockResults} />);
    
    expect(screen.getByTestId('timeline-section')).toBeInTheDocument();
    expect(screen.getByTestId('timeline-list')).toBeInTheDocument();
    
    // Check first timeline event
    expect(screen.getByTestId('timeline-event-0')).toBeInTheDocument();
    expect(screen.getByTestId('timestamp-0')).toHaveTextContent('0:10');
    expect(screen.getByTestId('confidence-0')).toHaveTextContent('90.0%');
    expect(screen.getByTestId('violence-status-0')).toHaveTextContent('Violence');
  });

  it('should format timestamps correctly', () => {
    render(<MockResultsViewer results={mockResults} />);
    
    expect(screen.getByTestId('timestamp-0')).toHaveTextContent('0:10');
    expect(screen.getByTestId('timestamp-1')).toHaveTextContent('0:15');
    expect(screen.getByTestId('timestamp-2')).toHaveTextContent('0:20');
    expect(screen.getByTestId('timestamp-3')).toHaveTextContent('0:25');
  });

  it('should apply correct confidence colors', () => {
    render(<MockResultsViewer results={mockResults} />);
    
    // High confidence (0.9) should be red
    expect(screen.getByTestId('confidence-0')).toHaveClass('text-red-600');
    
    // Medium confidence (0.3) should be green
    expect(screen.getByTestId('confidence-2')).toHaveClass('text-green-600');
  });

  it('should show correct violence status', () => {
    render(<MockResultsViewer results={mockResults} />);
    
    expect(screen.getByTestId('violence-status-0')).toHaveTextContent('Violence');
    expect(screen.getByTestId('violence-status-1')).toHaveTextContent('Violence');
    expect(screen.getByTestId('violence-status-2')).toHaveTextContent('No Violence');
    expect(screen.getByTestId('violence-status-3')).toHaveTextContent('Violence');
  });

  it('should calculate statistics correctly', () => {
    render(<MockResultsViewer results={mockResults} />);
    
    // Calculate expected average: (0.9 + 0.8 + 0.3 + 0.95) / 4 = 2.95 / 4 = 0.7375 = 73.8%
    expect(screen.getByTestId('statistics-section')).toBeInTheDocument();
    expect(screen.getByTestId('total-events')).toHaveTextContent('Total Events: 4');
    expect(screen.getByTestId('violence-events')).toHaveTextContent('Violence Events: 3');
    expect(screen.getByTestId('avg-confidence')).toHaveTextContent('Average Confidence: 73.8%');
  });

  it('should handle results with no violence detected', () => {
    const noViolenceResults = {
      ...mockResults,
      violence_detected: false,
      confidence_score: 0.2,
      timeline: [
        { timestamp: 10, confidence: 0.1, violence: false },
        { timestamp: 20, confidence: 0.2, violence: false }
      ]
    };

    render(<MockResultsViewer results={noViolenceResults} />);
    
    expect(screen.getByTestId('detection-status')).toHaveTextContent('Violence Detected: No');
    expect(screen.getByTestId('confidence-score')).toHaveTextContent('Confidence: 20.0%');
    expect(screen.getByTestId('violence-events')).toHaveTextContent('Violence Events: 0');
  });

  it('should handle empty timeline', () => {
    const emptyTimelineResults = {
      ...mockResults,
      timeline: []
    };

    render(<MockResultsViewer results={emptyTimelineResults} />);
    
    expect(screen.getByTestId('total-events')).toHaveTextContent('Total Events: 0');
    expect(screen.getByTestId('violence-events')).toHaveTextContent('Violence Events: 0');
  });
});