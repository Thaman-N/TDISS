import { useState, useEffect } from 'react';

// Theme definitions with proper HSL format
const themes = {
  default: {
    name: "Default",
    light: {
      '--background': '0 0% 100%',
      '--foreground': '240 10% 3.9%',
      '--card': '0 0% 100%',
      '--card-foreground': '240 10% 3.9%',
      '--popover': '0 0% 100%',
      '--popover-foreground': '240 10% 3.9%',
      '--primary': '240 5.9% 10%',
      '--primary-foreground': '0 0% 98%',
      '--secondary': '240 4.8% 95.9%',
      '--secondary-foreground': '240 5.9% 10%',
      '--muted': '240 4.8% 95.9%',
      '--muted-foreground': '240 3.8% 46.1%',
      '--accent': '240 4.8% 95.9%',
      '--accent-foreground': '240 5.9% 10%',
      '--destructive': '0 84.2% 60.2%',
      '--destructive-foreground': '0 0% 98%',
      '--border': '240 5.9% 90%',
      '--input': '240 5.9% 90%',
      '--ring': '240 10% 3.9%',
      '--radius': '0.5rem'
    },
    dark: {
      '--background': '240 10% 3.9%',
      '--foreground': '0 0% 98%',
      '--card': '240 10% 3.9%',
      '--card-foreground': '0 0% 98%',
      '--popover': '240 10% 3.9%',
      '--popover-foreground': '0 0% 98%',
      '--primary': '0 0% 98%',
      '--primary-foreground': '240 5.9% 10%',
      '--secondary': '240 3.7% 15.9%',
      '--secondary-foreground': '0 0% 98%',
      '--muted': '240 3.7% 15.9%',
      '--muted-foreground': '240 5% 64.9%',
      '--accent': '240 3.7% 15.9%',
      '--accent-foreground': '0 0% 98%',
      '--destructive': '0 62.8% 30.6%',
      '--destructive-foreground': '0 0% 98%',
      '--border': '240 3.7% 15.9%',
      '--input': '240 3.7% 15.9%',
      '--ring': '240 4.9% 83.9%',
      '--radius': '0.5rem'
    }
  },
  graphite: {
    name: "Graphite",
    light: {
      '--background': '0 0% 95.5%',
      '--foreground': '0 0% 32.1%',
      '--card': '0 0% 97%',
      '--card-foreground': '0 0% 32.1%',
      '--popover': '0 0% 97%',
      '--popover-foreground': '0 0% 32.1%',
      '--primary': '0 0% 48.9%',
      '--primary-foreground': '0 0% 100%',
      '--secondary': '0 0% 90.7%',
      '--secondary-foreground': '0 0% 32.1%',
      '--muted': '0 0% 88.5%',
      '--muted-foreground': '0 0% 51%',
      '--accent': '0 0% 80.8%',
      '--accent-foreground': '0 0% 32.1%',
      '--destructive': '26 86% 56%',
      '--destructive-foreground': '0 0% 100%',
      '--border': '0 0% 85.8%',
      '--input': '0 0% 90.7%',
      '--ring': '0 0% 48.9%',
      '--radius': '0.35rem'
    },
    dark: {
      '--background': '0 0% 21.8%',
      '--foreground': '0 0% 88.5%',
      '--card': '0 0% 24.4%',
      '--card-foreground': '0 0% 88.5%',
      '--popover': '0 0% 24.4%',
      '--popover-foreground': '0 0% 88.5%',
      '--primary': '0 0% 70.6%',
      '--primary-foreground': '0 0% 21.8%',
      '--secondary': '0 0% 30.9%',
      '--secondary-foreground': '0 0% 88.5%',
      '--muted': '0 0% 28.5%',
      '--muted-foreground': '0 0% 60%',
      '--accent': '0 0% 37.2%',
      '--accent-foreground': '0 0% 88.5%',
      '--destructive': '22 75% 66%',
      '--destructive-foreground': '0 0% 100%',
      '--border': '0 0% 32.9%',
      '--input': '0 0% 30.9%',
      '--ring': '0 0% 70.6%',
      '--radius': '0.35rem'
    }
  },
  claymorphism: {
    name: "Claymorphism",
    light: {
      '--background': '20 14% 90%',
      '--foreground': '217 32% 17%',
      '--card': '60 5% 96%',
      '--card-foreground': '217 32% 17%',
      '--popover': '60 5% 96%',
      '--popover-foreground': '217 32% 17%',
      '--primary': '239 83% 67%',
      '--primary-foreground': '0 0% 100%',
      '--secondary': '24 6% 83%',
      '--secondary-foreground': '215 14% 34%',
      '--muted': '20 14% 90%',
      '--muted-foreground': '220 9% 46%',
      '--accent': '293 44% 93%',
      '--accent-foreground': '217 19% 27%',
      '--destructive': '0 84% 60%',
      '--destructive-foreground': '0 0% 100%',
      '--border': '24 6% 83%',
      '--input': '24 6% 83%',
      '--ring': '239 83% 67%',
      '--radius': '1.25rem'
    },
    dark: {
      '--background': '30 11% 11%',
      '--foreground': '214 32% 91%',
      '--card': '26 9% 16%',
      '--card-foreground': '214 32% 91%',
      '--popover': '26 9% 16%',
      '--popover-foreground': '214 32% 91%',
      '--primary': '234 89% 74%',
      '--primary-foreground': '30 11% 11%',
      '--secondary': '26 6% 21%',
      '--secondary-foreground': '216 12% 84%',
      '--muted': '26 9% 16%',
      '--muted-foreground': '218 11% 65%',
      '--accent': '26 5% 27%',
      '--accent-foreground': '216 12% 84%',
      '--destructive': '0 84% 60%',
      '--destructive-foreground': '30 11% 11%',
      '--border': '26 6% 21%',
      '--input': '26 6% 21%',
      '--ring': '234 89% 74%',
      '--radius': '1.25rem'
    }
  },
  vercel: {
    name: "Vercel",
    light: {
      '--background': '0 0% 99%',
      '--foreground': '0 0% 0%',
      '--card': '0 0% 100%',
      '--card-foreground': '0 0% 0%',
      '--popover': '0 0% 99%',
      '--popover-foreground': '0 0% 0%',
      '--primary': '0 0% 0%',
      '--primary-foreground': '0 0% 100%',
      '--secondary': '0 0% 94%',
      '--secondary-foreground': '0 0% 0%',
      '--muted': '0 0% 97%',
      '--muted-foreground': '0 0% 44%',
      '--accent': '0 0% 94%',
      '--accent-foreground': '0 0% 0%',
      '--destructive': '23 89% 63%',
      '--destructive-foreground': '0 0% 100%',
      '--border': '0 0% 92%',
      '--input': '0 0% 94%',
      '--ring': '0 0% 0%',
      '--radius': '0.5rem'
    },
    dark: {
      '--background': '0 0% 0%',
      '--foreground': '0 0% 100%',
      '--card': '0 0% 14%',
      '--card-foreground': '0 0% 100%',
      '--popover': '0 0% 18%',
      '--popover-foreground': '0 0% 100%',
      '--primary': '0 0% 100%',
      '--primary-foreground': '0 0% 0%',
      '--secondary': '0 0% 25%',
      '--secondary-foreground': '0 0% 100%',
      '--muted': '0 0% 23%',
      '--muted-foreground': '0 0% 72%',
      '--accent': '0 0% 32%',
      '--accent-foreground': '0 0% 100%',
      '--destructive': '24 91% 69%',
      '--destructive-foreground': '0 0% 0%',
      '--border': '0 0% 26%',
      '--input': '0 0% 32%',
      '--ring': '0 0% 72%',
      '--radius': '0.5rem'
    }
  },
  twitter: {
    name: "Twitter",
    light: {
      '--background': '0 0% 100%',
      '--foreground': '210 25% 8%',
      '--card': '180 7% 97%',
      '--card-foreground': '210 25% 8%',
      '--popover': '0 0% 100%',
      '--popover-foreground': '210 25% 8%',
      '--primary': '204 88% 53%',
      '--primary-foreground': '0 0% 100%',
      '--secondary': '210 25% 8%',
      '--secondary-foreground': '0 0% 100%',
      '--muted': '240 2% 90%',
      '--muted-foreground': '210 25% 8%',
      '--accent': '212 51% 93%',
      '--accent-foreground': '204 88% 53%',
      '--destructive': '356 91% 54%',
      '--destructive-foreground': '0 0% 100%',
      '--border': '201 30% 91%',
      '--input': '200 23% 97%',
      '--ring': '203 89% 53%',
      '--radius': '1.3rem'
    },
    dark: {
      '--background': '0 0% 0%',
      '--foreground': '200 7% 91%',
      '--card': '228 10% 10%',
      '--card-foreground': '0 0% 85%',
      '--popover': '0 0% 0%',
      '--popover-foreground': '200 7% 91%',
      '--primary': '204 88% 53%',
      '--primary-foreground': '0 0% 100%',
      '--secondary': '195 15% 95%',
      '--secondary-foreground': '210 25% 8%',
      '--muted': '0 0% 9%',
      '--muted-foreground': '210 3% 46%',
      '--accent': '206 70% 8%',
      '--accent-foreground': '204 88% 53%',
      '--destructive': '356 91% 54%',
      '--destructive-foreground': '0 0% 100%',
      '--border': '210 5% 15%',
      '--input': '208 28% 18%',
      '--ring': '203 89% 53%',
      '--radius': '1.3rem'
    }
  },
  mocha: {
    name: "Mocha Mousse",
    light: {
      '--background': '55 30% 92%',
      '--foreground': '16 15% 29%',
      '--card': '55 30% 92%',
      '--card-foreground': '16 15% 29%',
      '--popover': '0 0% 100%',
      '--popover-foreground': '16 15% 29%',
      '--primary': '18 26% 52%',
      '--primary-foreground': '0 0% 100%',
      '--secondary': '38 22% 65%',
      '--secondary-foreground': '0 0% 100%',
      '--muted': '20 45% 81%',
      '--muted-foreground': '14 21% 45%',
      '--accent': '20 45% 81%',
      '--accent-foreground': '16 15% 29%',
      '--destructive': '23 15% 11%',
      '--destructive-foreground': '0 0% 100%',
      '--border': '38 22% 65%',
      '--input': '38 22% 65%',
      '--ring': '18 26% 52%',
      '--radius': '0.5rem'
    },
    dark: {
      '--background': '20 15% 15%',
      '--foreground': '55 30% 92%',
      '--card': '21 13% 21%',
      '--card-foreground': '55 30% 92%',
      '--popover': '21 13% 21%',
      '--popover-foreground': '55 30% 92%',
      '--primary': '22 33% 65%',
      '--primary-foreground': '20 15% 15%',
      '--secondary': '14 21% 45%',
      '--secondary-foreground': '55 30% 92%',
      '--muted': '16 15% 29%',
      '--muted-foreground': '21 27% 69%',
      '--accent': '38 22% 65%',
      '--accent-foreground': '20 15% 15%',
      '--destructive': '0 69% 67%',
      '--destructive-foreground': '20 15% 15%',
      '--border': '16 15% 29%',
      '--input': '16 15% 29%',
      '--ring': '22 33% 65%',
      '--radius': '0.5rem'
    }
  },
  supabase: {
    name: "Supabase",
    light: {
      '--background': '0 0% 99%',
      '--foreground': '0 0% 9%',
      '--card': '0 0% 99%',
      '--card-foreground': '0 0% 9%',
      '--popover': '0 0% 99%',
      '--popover-foreground': '0 0% 32%',
      '--primary': '151 67% 67%',
      '--primary-foreground': '153 13% 14%',
      '--secondary': '0 0% 99%',
      '--secondary-foreground': '0 0% 9%',
      '--muted': '0 0% 93%',
      '--muted-foreground': '0 0% 13%',
      '--accent': '0 0% 93%',
      '--accent-foreground': '0 0% 13%',
      '--destructive': '10 82% 44%',
      '--destructive-foreground': '0 100% 99%',
      '--border': '0 0% 87%',
      '--input': '0 0% 96%',
      '--ring': '151 67% 67%',
      '--radius': '0.5rem'
    },
    dark: {
      '--background': '0 0% 7%',
      '--foreground': '214 32% 91%',
      '--card': '0 0% 9%',
      '--card-foreground': '214 32% 91%',
      '--popover': '0 0% 14%',
      '--popover-foreground': '0 0% 66%',
      '--primary': '155 100% 19%',
      '--primary-foreground': '153 19% 89%',
      '--secondary': '0 0% 14%',
      '--secondary-foreground': '0 0% 98%',
      '--muted': '0 0% 12%',
      '--muted-foreground': '0 0% 64%',
      '--accent': '0 0% 19%',
      '--accent-foreground': '0 0% 98%',
      '--destructive': '7 60% 21%',
      '--destructive-foreground': '12 12% 92%',
      '--border': '0 0% 16%',
      '--input': '0 0% 14%',
      '--ring': '142 69% 58%',
      '--radius': '0.5rem'
    }
  }
};

const useTheme = () => {
  // Dark mode state
  const [darkMode, setDarkMode] = useState(() => {
    try {
      const saved = localStorage.getItem('darkMode');
      return saved !== null ? JSON.parse(saved) : true;
    } catch (error) {
      console.warn('localStorage not available, defaulting to dark mode:', error);
      return true;
    }
  });

  // Theme selection state
  const [currentTheme, setCurrentTheme] = useState(() => {
    try {
      const saved = localStorage.getItem('selectedTheme');
      return saved || 'default';
    } catch (error) {
      console.warn('localStorage not available, defaulting to default theme:', error);
      return 'default';
    }
  });

  // Apply theme + dark/light mode
  useEffect(() => {
    // Save preferences
    try {
      localStorage.setItem('darkMode', JSON.stringify(darkMode));
      localStorage.setItem('selectedTheme', currentTheme);
    } catch (error) {
      console.warn('Failed to save preferences:', error);
    }

    const root = document.documentElement;
    
    // Add theme-switching class to disable transitions temporarily
    root.classList.add('theme-switching');

    // Apply dark class
    if (darkMode) {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }

    // Apply theme variables - FORCE the update
    const themeVariables = themes[currentTheme]?.[darkMode ? 'dark' : 'light'];
    if (themeVariables) {
      // Clear existing custom properties first
      Object.keys(themes.default.light).forEach(property => {
        root.style.removeProperty(property);
      });
      
      // Apply new theme variables
      Object.entries(themeVariables).forEach(([property, value]) => {
        root.style.setProperty(property, value);
      });
    }
    
    // Remove theme-switching class after a short delay to re-enable transitions
    setTimeout(() => {
      root.classList.remove('theme-switching');
    }, 100);
  }, [darkMode, currentTheme]);

  return {
    darkMode,
    toggleDarkMode: () => setDarkMode(prev => !prev),
    setDarkMode,
    currentTheme,
    setTheme: (newTheme) => {
      if (themes[newTheme]) {
        setCurrentTheme(newTheme);
      }
    },
    availableThemes: Object.keys(themes),
    getThemeName: (themeKey) => themes[themeKey]?.name || themeKey
  };
};

export default useTheme;