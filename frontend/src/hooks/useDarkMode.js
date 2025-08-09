import { useState, useEffect } from 'react';

const useDarkMode = () => {
  // Initialize from localStorage or default to true (dark mode)
  const [darkMode, setDarkMode] = useState(() => {
    try {
      const saved = localStorage.getItem('darkMode');
      return saved !== null ? JSON.parse(saved) : true; // Default to dark mode
    } catch (error) {
      console.warn('localStorage not available, defaulting to dark mode:', error);
      return true; // Fallback to dark mode if localStorage fails
    }
  });

  // Apply dark mode to document and save to localStorage whenever darkMode changes
  useEffect(() => {
    // Save to localStorage
    try {
      localStorage.setItem('darkMode', JSON.stringify(darkMode));
    } catch (error) {
      console.warn('Failed to save dark mode preference to localStorage:', error);
    }

    // Apply to document
    if (darkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [darkMode]);

  // Toggle function
  const toggleDarkMode = () => {
    setDarkMode(prev => !prev);
  };

  return { 
    darkMode, 
    toggleDarkMode, 
    setDarkMode // In case manual setting is needed
  };
};

export default useDarkMode;