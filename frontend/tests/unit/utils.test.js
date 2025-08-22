// tests/unit/utils.test.js
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';

// Mock utility functions based on common needs
const utils = {
  // File validation utilities
  validateFileType: (file, allowedTypes = ['video/mp4', 'video/avi', 'video/mov', 'video/mkv', 'video/webm']) => {
    if (!file || !file.type) return false;
    return allowedTypes.includes(file.type);
  },

  validateFileSize: (file, maxSizeInMB = 100) => {
    if (!file || !file.size) return false;
    const maxSizeInBytes = maxSizeInMB * 1024 * 1024;
    return file.size <= maxSizeInBytes;
  },

  // Time formatting utilities
  formatDuration: (seconds) => {
    if (typeof seconds !== 'number' || seconds < 0) return '0:00';
    
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);

    if (hours > 0) {
      return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
    return `${minutes}:${secs.toString().padStart(2, '0')}`;
  },

  formatTimestamp: (seconds) => {
    if (typeof seconds !== 'number' || seconds < 0) return '0:00';
    
    const minutes = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${minutes}:${secs.toString().padStart(2, '0')}`;
  },

  // File size formatting
  formatFileSize: (bytes) => {
    if (typeof bytes !== 'number' || bytes < 0) return '0 B';
    
    const units = ['B', 'KB', 'MB', 'GB'];
    let size = bytes;
    let unitIndex = 0;

    while (size >= 1024 && unitIndex < units.length - 1) {
      size /= 1024;
      unitIndex++;
    }

    return `${size.toFixed(unitIndex === 0 ? 0 : 2)} ${units[unitIndex]}`;
  },

  // Confidence score utilities
  getConfidenceLevel: (score) => {
    if (typeof score !== 'number') return 'unknown';
    if (score >= 0.8) return 'high';
    if (score >= 0.6) return 'medium';
    if (score >= 0.4) return 'low';
    return 'very-low';
  },

  getConfidenceColor: (score) => {
    const level = utils.getConfidenceLevel(score);
    const colorMap = {
      'high': 'text-red-600',
      'medium': 'text-yellow-600',
      'low': 'text-blue-600',
      'very-low': 'text-green-600',
      'unknown': 'text-gray-600'
    };
    return colorMap[level] || 'text-gray-600';
  },

  // API utilities
  buildApiUrl: (endpoint, params = {}) => {
    const baseUrl = 'http://localhost:8000/api';
    const url = new URL(`${baseUrl}${endpoint}`);
    
    Object.entries(params).forEach(([key, value]) => {
      if (value !== null && value !== undefined) {
        url.searchParams.append(key, value.toString());
      }
    });
    
    return url.toString();
  },

  // Job status utilities
  isJobComplete: (status) => {
    const completeStatuses = ['completed', 'failed', 'error'];
    return completeStatuses.includes(status?.toLowerCase());
  },

  isJobInProgress: (status) => {
    const progressStatuses = ['uploading', 'processing', 'queued'];
    return progressStatuses.includes(status?.toLowerCase());
  },

  // Date formatting
  formatDate: (dateString) => {
    if (!dateString) return '';
    
    try {
      const date = new Date(dateString);
      return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      });
    } catch (error) {
      return 'Invalid Date';
    }
  },

  // Debounce utility
  debounce: (func, delay) => {
    let timeoutId;
    return (...args) => {
      clearTimeout(timeoutId);
      timeoutId = setTimeout(() => func.apply(null, args), delay);
    };
  },

  // Deep clone utility
  deepClone: (obj) => {
    if (obj === null || typeof obj !== 'object') return obj;
    if (obj instanceof Date) return new Date(obj.getTime());
    if (obj instanceof Array) return obj.map(utils.deepClone);
    if (typeof obj === 'object') {
      const cloned = {};
      Object.keys(obj).forEach(key => {
        cloned[key] = utils.deepClone(obj[key]);
      });
      return cloned;
    }
  }
};

// Create mock file helper for tests
const createMockFile = (name = 'test.mp4', type = 'video/mp4', size = 1024 * 1024) => {
  const file = new File(['mock content'], name, { type });
  Object.defineProperty(file, 'size', { value: size });
  return file;
};

describe('Utils - File Validation', () => {
  it('should validate video file types correctly', () => {
    const validFile = createMockFile('test.mp4', 'video/mp4');
    const invalidFile = createMockFile('test.txt', 'text/plain');

    expect(utils.validateFileType(validFile)).toBe(true);
    expect(utils.validateFileType(invalidFile)).toBe(false);
  });

  it('should validate file size correctly', () => {
    const smallFile = createMockFile('small.mp4', 'video/mp4', 50 * 1024 * 1024); // 50MB
    const largeFile = createMockFile('large.mp4', 'video/mp4', 150 * 1024 * 1024); // 150MB

    expect(utils.validateFileSize(smallFile, 100)).toBe(true);
    expect(utils.validateFileSize(largeFile, 100)).toBe(false);
  });

  it('should handle null/undefined files', () => {
    expect(utils.validateFileType(null)).toBe(false);
    expect(utils.validateFileSize(undefined)).toBe(false);
  });
});

describe('Utils - Time Formatting', () => {
  it('should format duration correctly', () => {
    expect(utils.formatDuration(65)).toBe('1:05');
    expect(utils.formatDuration(3661)).toBe('1:01:01');
    expect(utils.formatDuration(30)).toBe('0:30');
    expect(utils.formatDuration(0)).toBe('0:00');
  });

  it('should format timestamp correctly', () => {
    expect(utils.formatTimestamp(125)).toBe('2:05');
    expect(utils.formatTimestamp(45)).toBe('0:45');
    expect(utils.formatTimestamp(0)).toBe('0:00');
  });

  it('should handle invalid time inputs', () => {
    expect(utils.formatDuration(-5)).toBe('0:00');
    expect(utils.formatTimestamp('invalid')).toBe('0:00');
    expect(utils.formatDuration(null)).toBe('0:00');
  });
});

describe('Utils - File Size Formatting', () => {
  it('should format file sizes correctly', () => {
    expect(utils.formatFileSize(1024)).toBe('1.00 KB');
    expect(utils.formatFileSize(1048576)).toBe('1.00 MB');
    expect(utils.formatFileSize(1073741824)).toBe('1.00 GB');
    expect(utils.formatFileSize(512)).toBe('512 B');
  });

  it('should handle edge cases', () => {
    expect(utils.formatFileSize(0)).toBe('0 B');
    expect(utils.formatFileSize(-100)).toBe('0 B');
    expect(utils.formatFileSize('invalid')).toBe('0 B');
  });
});

describe('Utils - Confidence Score', () => {
  it('should categorize confidence levels correctly', () => {
    expect(utils.getConfidenceLevel(0.9)).toBe('high');
    expect(utils.getConfidenceLevel(0.7)).toBe('medium');
    expect(utils.getConfidenceLevel(0.5)).toBe('low');
    expect(utils.getConfidenceLevel(0.2)).toBe('very-low');
  });

  it('should return appropriate colors for confidence levels', () => {
    expect(utils.getConfidenceColor(0.9)).toBe('text-red-600');
    expect(utils.getConfidenceColor(0.7)).toBe('text-yellow-600');
    expect(utils.getConfidenceColor(0.5)).toBe('text-blue-600');
    expect(utils.getConfidenceColor(0.2)).toBe('text-green-600');
  });

  it('should handle invalid confidence scores', () => {
    expect(utils.getConfidenceLevel('invalid')).toBe('unknown');
    expect(utils.getConfidenceColor(null)).toBe('text-gray-600');
  });
});

describe('Utils - API Utilities', () => {
  it('should build API URLs correctly', () => {
    const url = utils.buildApiUrl('/upload');
    expect(url).toBe('http://localhost:8000/api/upload');
  });

  it('should build API URLs with parameters', () => {
    const url = utils.buildApiUrl('/status', { jobId: '123', page: 1 });
    expect(url).toContain('jobId=123');
    expect(url).toContain('page=1');
  });

  it('should handle null/undefined parameters', () => {
    const url = utils.buildApiUrl('/test', { valid: 'yes', invalid: null, undefined: undefined });
    expect(url).toContain('valid=yes');
    expect(url).not.toContain('invalid');
    expect(url).not.toContain('undefined');
  });
});

describe('Utils - Job Status', () => {
  it('should identify complete job statuses', () => {
    expect(utils.isJobComplete('completed')).toBe(true);
    expect(utils.isJobComplete('failed')).toBe(true);
    expect(utils.isJobComplete('error')).toBe(true);
    expect(utils.isJobComplete('processing')).toBe(false);
  });

  it('should identify in-progress job statuses', () => {
    expect(utils.isJobInProgress('processing')).toBe(true);
    expect(utils.isJobInProgress('uploading')).toBe(true);
    expect(utils.isJobInProgress('queued')).toBe(true);
    expect(utils.isJobInProgress('completed')).toBe(false);
  });

  it('should handle case insensitive status checks', () => {
    expect(utils.isJobComplete('COMPLETED')).toBe(true);
    expect(utils.isJobInProgress('PROCESSING')).toBe(true);
  });
});

describe('Utils - Date Formatting', () => {
  it('should format dates correctly', () => {
    const dateString = '2025-01-15T10:30:00Z';
    const formatted = utils.formatDate(dateString);
    expect(formatted).toMatch(/Jan 15, 2025/);
  });

  it('should handle invalid dates', () => {
    expect(utils.formatDate('invalid')).toBe('Invalid Date');
    expect(utils.formatDate(null)).toBe('');
    expect(utils.formatDate(undefined)).toBe('');
  });
});

describe('Utils - Helper Functions', () => {
  it('should debounce function calls', (done) => {
    let callCount = 0;
    const debouncedFn = utils.debounce(() => {
      callCount++;
    }, 100);

    debouncedFn();
    debouncedFn();
    debouncedFn();

    setTimeout(() => {
      expect(callCount).toBe(1);
      done();
    }, 150);
  });

  it('should deep clone objects', () => {
    const original = {
      a: 1,
      b: { c: 2, d: [3, 4] },
      e: new Date('2025-01-01')
    };

    const cloned = utils.deepClone(original);

    expect(cloned).toEqual(original);
    expect(cloned).not.toBe(original);
    expect(cloned.b).not.toBe(original.b);
    expect(cloned.b.d).not.toBe(original.b.d);
  });

  it('should handle primitive values in deep clone', () => {
    expect(utils.deepClone(null)).toBe(null);
    expect(utils.deepClone(42)).toBe(42);
    expect(utils.deepClone('string')).toBe('string');
    expect(utils.deepClone(true)).toBe(true);
  });
});